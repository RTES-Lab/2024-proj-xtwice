package com.example.useopencvwithcmakeandkotlin

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.view.MotionEvent
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import android.media.MediaMetadataRetriever

class HSVActivity : AppCompatActivity() {
    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }

    private lateinit var imageView: ImageView
    private lateinit var hsvInfoTextView: TextView
    private var matInput: Mat? = null
    private var roiData: ROIData? = null
    private var originalBitmap: Bitmap? = null
    private var touchedBitmap: Bitmap? = null
    private var touchedCanvas: Canvas? = null
    private val pointPaint = Paint().apply {
        color = Color.RED  // 빨간색 점
        style = Paint.Style.FILL
        strokeWidth = 10f
    }
    private var lastTouchX = -1f
    private var lastTouchY = -1f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_hsv)

        imageView = findViewById(R.id.imageView)
        hsvInfoTextView = findViewById(R.id.hsvInfoTextView)
        
        roiData = intent.getParcelableExtra("roiData")
        val videoUri = intent.getStringExtra("videoUri")?.let { Uri.parse(it) }
        
        loadAndProcessImage(videoUri)
        setupTouchListener()
    }

    private fun loadAndProcessImage(videoUri: Uri?) {
        videoUri?.let {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this, videoUri)
                originalBitmap = retriever.getFrameAtTime(0)
                
                roiData?.let { roi ->
                    val croppedBitmap = Bitmap.createBitmap(
                        originalBitmap!!,
                        roi.left,
                        roi.top,
                        roi.right - roi.left,
                        roi.bottom - roi.top
                    )
                    
                    touchedBitmap = croppedBitmap.copy(Bitmap.Config.ARGB_8888, true)
                    touchedCanvas = Canvas(touchedBitmap!!)
                    
                    matInput = Mat()
                    Utils.bitmapToMat(croppedBitmap, matInput)
                    imageView.setImageBitmap(touchedBitmap)
                }
            } finally {
                retriever.release()
            }
        }
    }

    private fun setupTouchListener() {
        imageView.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                    try {
                        // 이미지뷰 내에서의 실제 이미지 위치와 크기를 계산
                        val imageView = view as ImageView
                        val drawable = imageView.drawable
                        if (drawable == null) return@setOnTouchListener false
                        
                        // 이미지뷰의 실제 크기
                        val imageViewWidth = imageView.width - imageView.paddingLeft - imageView.paddingRight
                        val imageViewHeight = imageView.height - imageView.paddingTop - imageView.paddingBottom
                        
                        // 실제 이미지의 크기
                        val imageWidth = drawable.intrinsicWidth
                        val imageHeight = drawable.intrinsicHeight
                        
                        // 이미지가 이미지뷰에서 실제로 그려지는 크기와 위치를 계산
                        val scale: Float
                        var offsetX = imageView.paddingLeft.toFloat()
                        var offsetY = imageView.paddingTop.toFloat()
                        
                        if (imageWidth * imageViewHeight > imageViewWidth * imageHeight) {
                            // 이미지가 가로로 더 길 경우
                            scale = imageViewWidth.toFloat() / imageWidth.toFloat()
                            offsetY += (imageViewHeight - imageHeight * scale) / 2.0f
                        } else {
                            // 이미지가 세로로 더 길 경우
                            scale = imageViewHeight.toFloat() / imageHeight.toFloat()
                            offsetX += (imageViewWidth - imageWidth * scale) / 2.0f
                        }
                        
                        // 터치 좌표를 실제 이미지 좌표로 변환
                        val x = ((event.x - offsetX) / scale).toInt()
                        val y = ((event.y - offsetY) / scale).toInt()
                        
                        // 좌표가 이미지 범위 내에 있는지 확인
                        if (x < 0 || x >= imageWidth || y < 0 || y >= imageHeight) {
                            return@setOnTouchListener true
                        }
                        
                        touchedBitmap?.let { bitmap ->
                            touchedCanvas?.drawBitmap(matInput?.let { mat ->
                                val tempBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
                                Utils.matToBitmap(mat, tempBitmap)
                                tempBitmap
                            } ?: return@let, 0f, 0f, null)
                            
                            touchedCanvas?.drawCircle(x.toFloat(), y.toFloat(), 5f, pointPaint)
                            imageView.setImageBitmap(touchedBitmap)
                            
                            lastTouchX = x.toFloat()
                            lastTouchY = y.toFloat()
                        }

                        matInput?.let { mat ->
                            val hsvMat = Mat()
                            Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_BGR2HSV)
                            
                            val pixel = ByteArray(3)
                            hsvMat.get(y, x, pixel)
                            
                            val hue = pixel[0].toInt() and 0xFF
                            val saturation = pixel[1].toInt() and 0xFF
                            val value = pixel[2].toInt() and 0xFF

                            val hueAngle = (hue * 2)
                            val saturationPercent = (saturation * 100 / 255)
                            val valuePercent = (value * 100 / 255)
                            
                            hsvInfoTextView.text = """
                                원본 HSV: ($hue, $saturation, $value)
                                변환 HSV: (${hueAngle}°, ${saturationPercent}%, ${valuePercent}%)
                                색상: ${getColorName(hue, saturation, value)}
                            """.trimIndent()
                            
                            hsvMat.release()
                        }
                    } catch (e: Exception) {
                        e.printStackTrace()
                    }
                    true
                }
                else -> false
            }
        }
    }

    private fun getColorName(hue: Int, saturation: Int, value: Int): String {
        if (value < 30) {
            return "검정"
        }
        if (saturation < 50) {
            return if (value < 128) "어두운 회색" else "밝은 회색"
        }
        if (value > 240 && saturation < 80) {
            return "흰색"
        }

        return when (hue) {
            in 0..9 -> "빨강"
            in 10..20 -> "주황"
            in 21..35 -> "노랑"
            in 36..75 -> "초록"
            in 76..100 -> "청록"
            in 101..125 -> "파랑"
            in 126..145 -> "보라"
            in 146..179 -> "빨강"
            else -> "알 수 없음"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        matInput?.release()
        touchedBitmap?.recycle()
        touchedBitmap = null
    }
}