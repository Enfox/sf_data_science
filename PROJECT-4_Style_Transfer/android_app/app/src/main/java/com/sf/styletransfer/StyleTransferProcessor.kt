package com.sf.styletransfer

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class StyleTransferProcessor(private val context: Context) {

    private var encoder: Interpreter? = null
    private var decoder: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null

    private val styleMeans = mutableMapOf<String, FloatArray>()
    private val styleStds = mutableMapOf<String, FloatArray>()
    private var currentStyle: String? = null

    companion object {
        private const val TAG = "StyleTransfer"
        private const val IMG = 256
        private const val FS = 32         // feature spatial size
        private const val FC = 512        // feature channels
        private const val ENCODER = "encoder_relu4_1.tflite"
        private const val DECODER = "decoder.tflite"
        private const val SV = "style_vectors"
        private const val SJ = "$SV/styles.json"
    }

    fun initialize() {
        Log.i(TAG, "Initializing models...")

        // CPU-only first (GPU delegate crashes with REFLECT pad on many devices)
        val encOpts = Interpreter.Options().setNumThreads(4)
        val decOpts = Interpreter.Options().setNumThreads(4)

        encoder = Interpreter(loadModelFile(ENCODER), encOpts)
        decoder = Interpreter(loadModelFile(DECODER), decOpts)
        Log.i(TAG, "Models loaded OK")

        loadStyleVectors()
        Log.i(TAG, "Loaded ${styleMeans.size} styles")
    }

    private fun loadStyleVectors() {
        val json = context.assets.open(SJ).bufferedReader().use { it.readText() }
        val styles = JSONArray(json)
        for (i in 0 until styles.length()) {
            val name = styles.getString(i)
            try {
                styleMeans[name] = loadNpyFloats(context.assets.open("$SV/${name}_mean.npy"))
                styleStds[name] = loadNpyFloats(context.assets.open("$SV/${name}_std.npy"))
            } catch (e: Exception) {
                Log.w(TAG, "Failed to load style '$name': ${e.message}")
            }
        }
        currentStyle = styleMeans.keys.firstOrNull()
    }

    private fun loadNpyFloats(stream: java.io.InputStream): FloatArray {
        val data = stream.use { it.readBytes() }
        val majorVer = data[6].toInt() and 0xFF
        var hdrStart: Int
        var hLen: Int
        if (majorVer >= 2) {
            hLen = ByteBuffer.wrap(data, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
            hdrStart = 12
        } else {
            hLen = ByteBuffer.wrap(data, 8, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xFFFF
            hdrStart = 10
        }
        val dataStart = hdrStart + hLen
        val n = (data.size - dataStart) / 4
        val result = FloatArray(n)
        ByteBuffer.wrap(data, dataStart, n * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result)
        return result
    }

    fun getAvailableStyles(): List<String> = styleMeans.keys.toList()

    fun setStyle(name: String) {
        if (styleMeans.containsKey(name)) currentStyle = name
    }

    fun getCurrentStyle(): String? = currentStyle

    /**
     * Main style transfer: content bitmap → encode → AdaIN → decode → stylized bitmap.
     * Uses multi-dimensional arrays matching TFLite model shapes.
     */
    fun stylize(contentBitmap: Bitmap): Bitmap? {
        val styleName = currentStyle ?: return null
        val sMean = styleMeans[styleName] ?: return null
        val sStd = styleStds[styleName] ?: return null
        val enc = encoder ?: return null
        val dec = decoder ?: return null

        try {
            // 1. Preprocess: [1,256,256,3] NHWC multi-dim array
            val resized = Bitmap.createScaledBitmap(contentBitmap, IMG, IMG, true)
            val cInput = Array(IMG) { Array(IMG) { FloatArray(3) } }
            val px = IntArray(IMG * IMG)
            resized.getPixels(px, 0, IMG, 0, 0, IMG, IMG)
            for (i in 0 until IMG) {
                for (j in 0 until IMG) {
                    val p = px[i * IMG + j]
                    cInput[i][j][0] = ((p shr 16) and 0xFF) / 255f
                    cInput[i][j][1] = ((p shr 8) and 0xFF) / 255f
                    cInput[i][j][2] = (p and 0xFF) / 255f
                }
            }
            val contentInput = arrayOf(cInput)  // [1][256][256][3]

            // 2. Encode: → [1,32,32,512] NHWC
            val feat = Array(1) { Array(FS) { Array(FS) { FloatArray(FC) } } }
            enc.run(contentInput, feat)

            // 3. AdaIN (compute on flat indices)
            // feat shape: [1][FS][FS][FC] — Kotlin sees Array<Array<Array<FloatArray>>>
            val fBatch = feat[0]  // [FS][FS][FC]
            val meanC = FloatArray(FC)
            val stdC = FloatArray(FC)
            val n = FS * FS

            // mean per channel
            for (h in 0 until FS) {
                for (w in 0 until FS) {
                    val row = fBatch[h][w]
                    for (c in 0 until FC) meanC[c] += row[c]
                }
            }
            for (c in 0 until FC) meanC[c] /= n.toFloat()

            // std per channel
            for (h in 0 until FS) {
                for (w in 0 until FS) {
                    val row = fBatch[h][w]
                    for (c in 0 until FC) {
                        val d = row[c] - meanC[c]
                        stdC[c] += d * d
                    }
                }
            }
            for (c in 0 until FC) stdC[c] = sqrt(stdC[c] / n + 1e-5f)

            // apply AdaIN
            for (h in 0 until FS) {
                for (w in 0 until FS) {
                    val row = fBatch[h][w]
                    for (c in 0 until FC) {
                        val norm = (row[c] - meanC[c]) / stdC[c]
                        row[c] = sStd[c] * norm + sMean[c]
                    }
                }
            }

            // 4. Decode: [1,32,32,512] → [1,256,256,3]
            val outImg = Array(1) { Array(IMG) { Array(IMG) { FloatArray(3) } } }
            dec.run(feat, outImg)

            // 5. Postprocess → Bitmap
            val pixels = IntArray(IMG * IMG)
            val o = outImg[0]
            for (i in 0 until IMG) {
                for (j in 0 until IMG) {
                    val r = clamp(o[i][j][0] * 255f)
                    val g = clamp(o[i][j][1] * 255f)
                    val b = clamp(o[i][j][2] * 255f)
                    pixels[i * IMG + j] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            val bmp = Bitmap.createBitmap(IMG, IMG, Bitmap.Config.ARGB_8888)
            bmp.setPixels(pixels, 0, IMG, 0, 0, IMG, IMG)
            return bmp

        } catch (e: Exception) {
            Log.e(TAG, "stylize() error: ${e.message}", e)
            return null
        }
    }

    private fun clamp(v: Float): Int = min(255, max(0, v.toInt()))

    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val fd = context.assets.openFd(fileName)
        val fis = FileInputStream(fd.fileDescriptor)
        val ch = fis.channel
        return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }

    fun close() {
        encoder?.close()
        decoder?.close()
        gpuDelegate?.close()
        encoder = null
        decoder = null
        gpuDelegate = null
    }
}
