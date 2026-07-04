package com.sf.styletransfer

import android.Manifest
import android.annotation.SuppressLint
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import kotlinx.coroutines.*
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var resultImage: ImageView
    private lateinit var captureButton: ImageButton
    private lateinit var backButton: ImageButton
    private lateinit var saveButton: Button
    private lateinit var transferButton: Button
    private lateinit var stylesRecycler: RecyclerView
    private lateinit var loadingBar: ProgressBar
    private lateinit var inferenceTimeText: TextView
    private lateinit var cameraContainer: View
    private lateinit var resultContainer: View

    private var processor: StyleTransferProcessor? = null
    private var cameraExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var imageCapture: ImageCapture? = null
    private var capturedBitmap: Bitmap? = null
    private var selectedStylePos = 0

    companion object {
        private const val CAMERA_PERMISSION = Manifest.permission.CAMERA
        private const val REQUEST_CODE = 1001
        private const val TAG = "StyleTransfer"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        resultImage = findViewById(R.id.resultImage)
        captureButton = findViewById(R.id.captureButton)
        backButton = findViewById(R.id.backButton)
        saveButton = findViewById(R.id.saveButton)
        transferButton = findViewById(R.id.transferButton)
        stylesRecycler = findViewById(R.id.stylesRecycler)
        loadingBar = findViewById(R.id.loadingBar)
        inferenceTimeText = findViewById(R.id.inferenceTimeText)
        cameraContainer = findViewById(R.id.cameraContainer)
        resultContainer = findViewById(R.id.resultContainer)

        captureButton.setOnClickListener { takePhoto() }
        backButton.setOnClickListener { showCamera() }
        saveButton.setOnClickListener { saveResult() }
        transferButton.setOnClickListener { applyStyle() }

        showCamera()

        // Load model in background
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                processor = StyleTransferProcessor(this@MainActivity).apply { initialize() }
                withContext(Dispatchers.Main) {
                    setupStyles()
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model init failed", e)
            }
        }
    }

    private fun setupStyles() {
        val styles = processor?.getAvailableStyles() ?: return
        val styleImages = styles.mapIndexed { idx, name ->
            StyleItem(name, getStyleThumbnail(name))
        }
        val adapter = StyleAdapter(styleImages) { pos ->
            selectedStylePos = pos
            processor?.setStyle(styles[pos])
        }
        adapter.select(0)
        if (styles.isNotEmpty()) processor?.setStyle(styles[0])

        stylesRecycler.layoutManager = LinearLayoutManager(this, LinearLayoutManager.HORIZONTAL, false)
        stylesRecycler.adapter = adapter
    }

    private fun getStyleThumbnail(name: String): Bitmap? {
        return try {
            val files = assets.list("styles") ?: return null
            val match = files.find { it.startsWith(name) } ?: return null
            assets.open("styles/$match").use { BitmapFactory.decodeStream(it) }
        } catch (e: Exception) { null }
    }

    // ─── Camera ───────────────────────────────────────────

    private fun showCamera() {
        cameraContainer.visibility = View.VISIBLE
        resultContainer.visibility = View.GONE
        if (hasCameraPermission()) startCamera()
        else requestCameraPermission()
    }

    private fun startCamera() {
        val providerFuture = ProcessCameraProvider.getInstance(this)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                .setTargetRotation(windowManager.defaultDisplay.rotation)
                .build()
            val selector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                provider.unbindAll()
                provider.bindToLifecycle(this, selector, preview, imageCapture)
            } catch (e: Exception) {
                Log.e(TAG, "Camera bind failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("MissingPermission")
    private fun takePhoto() {
        val ic = imageCapture ?: run {
            Toast.makeText(this, "Camera not ready", Toast.LENGTH_SHORT).show()
            return
        }

        ic.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    // Convert ImageProxy to Bitmap safely
                    val bmp = image.toBitmap()
                    image.close()

                    // Rotate based on sensor orientation
                    val matrix = Matrix()
                    matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())

                    val rotated = Bitmap.createBitmap(
                        bmp, 0, 0, bmp.width, bmp.height, matrix, true
                    )

                    capturedBitmap = rotated
                    runOnUiThread {
                        resultImage.setImageBitmap(rotated)
                        showResult()
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Image processing error", e)
                    image.close()
                    runOnUiThread {
                        Toast.makeText(
                            this@MainActivity,
                            "Capture error: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }

            override fun onError(exc: ImageCaptureException) {
                Log.e(TAG, "Capture failed", exc)
                runOnUiThread {
                    Toast.makeText(
                        this@MainActivity,
                        "Camera error: ${exc.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        })
    }

    // ─── Result ──────────────────────────────────────────

    private fun showResult() {
        cameraContainer.visibility = View.GONE
        resultContainer.visibility = View.VISIBLE
    }

    private fun applyStyle() {
        val bmp = capturedBitmap ?: return
        val proc = processor ?: return

        loadingBar.visibility = View.VISIBLE
        transferButton.isEnabled = false
        inferenceTimeText.text = "Processing..."

        lifecycleScope.launch(Dispatchers.Default) {
            val startTime = System.currentTimeMillis()
            val result = proc.stylize(bmp)
            val elapsed = System.currentTimeMillis() - startTime

            withContext(Dispatchers.Main) {
                loadingBar.visibility = View.GONE
                transferButton.isEnabled = true
                if (result != null) {
                    resultImage.setImageBitmap(result)
                    inferenceTimeText.text = "${elapsed}ms"
                } else {
                    Toast.makeText(this@MainActivity, "Style transfer failed", Toast.LENGTH_SHORT).show()
                    inferenceTimeText.text = "Error"
                }
            }
        }
    }

    private fun saveResult() {
        val drawable = resultImage.drawable ?: return
        val bmp = (drawable as android.graphics.drawable.BitmapDrawable).bitmap
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME,
                        "stylized_${SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())}.jpg")
                    put(MediaStore.MediaColumns.MIME_TYPE, "image/jpeg")
                    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q)
                        put(MediaStore.MediaColumns.RELATIVE_PATH, "Pictures/StyleTransfer")
                }
                val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    contentResolver.openOutputStream(uri)?.use { os: OutputStream ->
                        bmp.compress(Bitmap.CompressFormat.JPEG, 95, os)
                    }
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@MainActivity, "Saved!", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Save failed", e)
            }
        }
    }

    // ─── Permissions ─────────────────────────────────────

    private fun hasCameraPermission() =
        ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(this, arrayOf(CAMERA_PERMISSION), REQUEST_CODE)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)
            startCamera()
        else
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        processor?.close()
    }
}

// ─── Style data + adapter ──────────────────────────────────

data class StyleItem(val name: String, val thumbnail: Bitmap?)

class StyleAdapter(
    private val styles: List<StyleItem>,
    private val onClick: (Int) -> Unit
) : RecyclerView.Adapter<StyleAdapter.VH>() {

    private var selected = 0

    fun select(pos: Int) { selected = pos; notifyDataSetChanged() }

    inner class VH(val view: View) : RecyclerView.ViewHolder(view) {
        val img: ImageView = view.findViewById(R.id.styleThumb)
        val sel: View = view.findViewById(R.id.styleSelected)
    }

    override fun onCreateViewHolder(parent: android.view.ViewGroup, viewType: Int): VH {
        val v = android.view.LayoutInflater.from(parent.context)
            .inflate(R.layout.item_style, parent, false)
        return VH(v)
    }

    override fun onBindViewHolder(holder: VH, position: Int) {
        val s = styles[position]
        if (s.thumbnail != null) holder.img.setImageBitmap(s.thumbnail)
        else holder.img.setBackgroundColor(0xFF6200EE.toInt())
        holder.sel.visibility = if (position == selected) View.VISIBLE else View.INVISIBLE
        holder.view.setOnClickListener {
            selected = position
            notifyDataSetChanged()
            onClick(position)
        }
    }

    override fun getItemCount() = styles.size
}
