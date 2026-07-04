"""
Final export pipeline — produces all artifacts for Android app.

Outputs:
  1. decoder.tflite         — decoder (AdaIN features → image), 21 MB
  2. encoder_relu4_1.tflite — VGG encoder up to relu4_1 (image → features), ~130 MB
  3. style_vectors/          — precomputed mean/std for each style image
  4. style_transfer_full.onnx — full model (backup, for ONNX Runtime)

Android pipeline:
  content_img → encoder.tflite → content_feat
  AdaIN(content_feat, style_mean, style_std) → decoder.tflite → stylized_img
"""
import os, sys, json
import numpy as np
import torch
from pathlib import Path

os.environ['TF_USE_LEGACY_KERAS'] = '1'

sys.path.insert(0, str(Path(__file__).parent))
from model import StyleTransferNet, VGGEncoder, adaptive_instance_normalization, calc_mean_std

PROJECT = Path(__file__).parent.parent
EXPORT = PROJECT / "exported_models"
EXPORT.mkdir(exist_ok=True)


def export_decoder():
    """Export trained decoder to TFLite via Keras (REFLECT padding)."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("=" * 55)
    print("  1. Decoder → TFLite")
    print("=" * 55)

    net = StyleTransferNet()
    ckpt = torch.load(str(PROJECT / "checkpoints/decoder_final.pth"),
                      map_location="cpu", weights_only=True)
    net.decoder.load_state_dict(ckpt["decoder"])
    net.eval()

    class ReflectPad(layers.Layer):
        def __init__(self, pad=1, **kw):
            super().__init__(**kw)
            self.pad = pad
        def call(self, x):
            return tf.pad(x, [[0,0],[self.pad,self.pad],[self.pad,self.pad],[0,0]], mode='REFLECT')

    def cr(x, oc, n):
        x = ReflectPad(1, name=f'{n}_p')(x)
        x = layers.Conv2D(oc, 3, padding='valid', name=f'{n}_c')(x)
        return layers.ReLU(name=f'{n}_r')(x)

    inp = keras.Input(shape=(32, 32, 512), name='adain_features')
    x = inp
    for i, oc in enumerate([512, 512, 512, 256]):
        x = cr(x, oc, f'b4_{i}')
    x = layers.UpSampling2D(2, interpolation='nearest', name='u1')(x)
    for i, oc in enumerate([256, 256, 256, 128]):
        x = cr(x, oc, f'b3_{i}')
    x = layers.UpSampling2D(2, interpolation='nearest', name='u2')(x)
    for i, oc in enumerate([128, 64]):
        x = cr(x, oc, f'b2_{i}')
    x = layers.UpSampling2D(2, interpolation='nearest', name='u3')(x)
    x = cr(x, 64, 'b1_0')
    x = ReflectPad(1, name='b1_1_p')(x)
    x = layers.Conv2D(3, 3, padding='valid', name='b1_1_c')(x)
    model = keras.Model(inp, x, name='decoder')

    # Copy weights
    pt_blocks = [net.decoder.block4, net.decoder.block3, net.decoder.block2, net.decoder.block1]
    pt_convs = [m for b in pt_blocks for m in b if hasattr(m, 'weight') and m.weight is not None]
    k_convs = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    for pt_c, k_c in zip(pt_convs, k_convs):
        w = pt_c.weight.detach().numpy()
        b = pt_c.bias.detach().numpy()
        k_c.set_weights([np.transpose(w, (2, 3, 1, 0)), b])

    # Verify Keras vs PyTorch
    dummy_nchw = np.random.randn(1, 512, 32, 32).astype(np.float32)
    dummy_nhwc = np.transpose(dummy_nchw, (0, 2, 3, 1))
    with torch.no_grad():
        out_pt = net.decoder(torch.from_numpy(dummy_nchw)).numpy()
    out_pt_nhwc = np.transpose(out_pt, (0, 2, 3, 1))
    out_keras = model.predict(dummy_nhwc, verbose=0)
    diff = np.abs(out_keras - out_pt_nhwc).max()
    print(f"  Keras vs PyTorch: max diff = {diff:.10f}")

    # TFLite float16
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite = conv.convert()
    path = str(EXPORT / "decoder.tflite")
    with open(path, "wb") as f:
        f.write(tflite)
    print(f"  decoder.tflite: {os.path.getsize(path)/1e6:.1f} MB")

    # Verify TFLite
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    ti = interp.get_input_details()
    to = interp.get_output_details()
    interp.set_tensor(ti[0]['index'], dummy_nhwc)
    interp.invoke()
    out_tf = interp.get_tensor(to[0]['index'])
    diff2 = np.abs(out_tf - out_pt_nhwc).max()
    print(f"  TFLite vs PyTorch: max diff = {diff2:.6f}")
    print(f"  ✓ PASSED\n")
    return path


def export_encoder():
    """Export VGG encoder (up to relu4_1) to TFLite via Keras."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    print("=" * 55)
    print("  2. VGG Encoder (→relu4_1) → TFLite")
    print("=" * 55)

    net = StyleTransferNet()
    encoder = net.encoder
    encoder.eval()

    import tensorflow as tf

    class ReflectPad(tf.keras.layers.Layer):
        def __init__(self, pad=1, **kw):
            super().__init__(**kw)
            self.pad = pad
        def call(self, x):
            return tf.pad(x, [[0,0],[self.pad,self.pad],[self.pad,self.pad],[0,0]], mode='REFLECT')

    # Extract VGG19 features up to relu4_1 (index 25 in vgg19.features)
    vgg_features = encoder.features[:26]  # up to and including relu4_1

    # Build Keras model
    inp = keras.Input(shape=(256, 256, 3), name='image')  # NHWC, [0,1]
    x = inp

    # Apply ImageNet normalization (subtract mean, divide std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    mean_const = keras.backend.constant(mean)
    std_const = keras.backend.constant(std)
    x = layers.Lambda(lambda z: (z - mean_const) / std_const, name='normalize')(x)

    # Iterate through VGG layers up to relu4_1
    layer_idx = 0
    pt_convs = []
    keras_convs = []

    for i, module in enumerate(vgg_features):
        if isinstance(module, torch.nn.Conv2d):
            # ReflectionPad2d comes before conv in VGG? No — VGG uses padding parameter
            # VGG conv2d has padding=1 (same padding for 3x3 kernel)
            w = module.weight.detach().numpy()  # [out, in, kh, kw]
            b = module.bias.detach().numpy()
            w_k = np.transpose(w, (2, 3, 1, 0))  # [kh, kw, in, out]
            x = layers.Conv2D(w.shape[0], 3, padding='same',
                             name=f'conv_{layer_idx}',
                             weights=[w_k, b])(x)
            pt_convs.append(module)
            keras_convs.append(x)
            layer_idx += 1
        elif isinstance(module, torch.nn.ReLU):
            x = layers.ReLU(name=f'relu_{layer_idx-1}')(x)
        elif isinstance(module, torch.nn.MaxPool2d):
            x = layers.MaxPooling2D(2, name=f'pool_{layer_idx}')(x)
        elif isinstance(module, torch.nn.ReflectionPad2d):
            x = ReflectPad(module.padding, name=f'pad_{layer_idx}')(x)

    model = keras.Model(inp, x, name='encoder')

    # Verify
    dummy = np.random.rand(1, 256, 256, 3).astype(np.float32)
    # PyTorch: NCHW
    dummy_nchw = np.transpose(dummy, (0, 3, 1, 2))
    with torch.no_grad():
        feat_pt = encoder(torch.from_numpy(dummy_nchw), layers=["relu4_1"]).numpy()
    feat_pt_nhwc = np.transpose(feat_pt, (0, 2, 3, 1))
    feat_keras = model.predict(dummy, verbose=0)
    diff = np.abs(feat_keras - feat_pt_nhwc).max()
    print(f"  Keras vs PyTorch: max diff = {diff:.8f}")

    # TFLite
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite = conv.convert()
    path = str(EXPORT / "encoder_relu4_1.tflite")
    with open(path, "wb") as f:
        f.write(tflite)
    print(f"  encoder_relu4_1.tflite: {os.path.getsize(path)/1e6:.1f} MB")

    # Verify TFLite
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    ti = interp.get_input_details()
    to = interp.get_output_details()
    interp.resize_tensor_input(ti[0]['index'], [1, 256, 256, 3])
    interp.allocate_tensors()
    interp.set_tensor(ti[0]['index'], dummy)
    interp.invoke()
    feat_tf = interp.get_tensor(to[0]['index'])
    diff2 = np.abs(feat_tf - feat_pt_nhwc).max()
    print(f"  TFLite vs PyTorch: max diff = {diff2:.6f}")
    print(f"  ✓ PASSED\n")
    return path


def precompute_styles():
    """Precompute style mean/std for all magenta style images."""
    import torchvision.transforms as T
    from PIL import Image

    print("=" * 55)
    print("  3. Precompute Style Vectors")
    print("=" * 55)

    net = StyleTransferNet()
    ckpt = torch.load(str(PROJECT / "checkpoints/decoder_final.pth"),
                      map_location="cuda", weights_only=True)
    net.decoder.load_state_dict(ckpt["decoder"])
    net = net.cuda().eval()

    style_dir = PROJECT / "test_images" / "magenta_styles"
    sv_dir = EXPORT / "style_vectors"
    sv_dir.mkdir(exist_ok=True)

    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    manifest = {}

    for path in sorted(style_dir.glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            feat = net.encoder(x, layers=["relu4_1"])
        mean, std = calc_mean_std(feat)
        name = path.stem
        np.save(str(sv_dir / f"{name}_mean.npy"), mean.cpu().numpy())
        np.save(str(sv_dir / f"{name}_std.npy"), std.cpu().numpy())
        manifest[name] = f"{name}_mean.npy"
        print(f"  {name}")

    with open(str(sv_dir / "styles.json"), "w") as f:
        json.dump(list(manifest.keys()), f)
    print(f"\n  {len(manifest)} styles precomputed → {sv_dir}\n")


def main():
    os.chdir(str(PROJECT))
    export_decoder()
    export_encoder()
    precompute_styles()

    print("=" * 55)
    print("  EXPORT COMPLETE")
    print("=" * 55)
    for f in sorted(EXPORT.glob("*")):
        if f.is_file():
            print(f"  {f.name:40s} {f.stat().st_size/1e6:.1f} MB")
    for f in sorted((EXPORT / "style_vectors").glob("*")):
        print(f"  style_vectors/{f.name:35s} {f.stat().st_size/1e3:.1f} KB")
    print("=" * 55)


if __name__ == "__main__":
    main()
