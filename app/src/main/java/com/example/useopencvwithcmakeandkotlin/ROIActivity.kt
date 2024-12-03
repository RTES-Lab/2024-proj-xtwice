package com.example.useopencvwithcmakeandkotlin

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Parcel
import android.os.Parcelable
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc

class ROIActivity : AppCompatActivity() {
    companion object {
        // OpenCV 라이브러리 로드
        init {
            System.loadLibrary("opencv_java4")
        }
    }

    private lateinit var imageView: ImageView
    private lateinit var cropButton: Button
    private lateinit var resetButton: Button
    private lateinit var finishROIButton: Button
    private var isROICropped = false
    private var matInput: Mat? = null
    private var startX = 0f
    private var startY = 0f
    private var currentRect: RectF? = null
    private var originalBitmap: Bitmap? = null
    private var drawingBitmap: Bitmap? = null
    private var canvas: Canvas? = null
    private val paint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 5f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_roi)

        imageView = findViewById(R.id.imageView)
        cropButton = findViewById(R.id.cropButton)
        resetButton = findViewById(R.id.resetButton)
        finishROIButton = findViewById(R.id.finishROIButton)
        cropButton.isEnabled = false

        val videoUri = intent.getStringExtra("videoUri")?.let { Uri.parse(it) }

        videoUri?.let {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this, videoUri)
                originalBitmap = retriever.getFrameAtTime(0)
                drawingBitmap = originalBitmap?.copy(Bitmap.Config.ARGB_8888, true)
                canvas = drawingBitmap?.let { Canvas(it) }

                matInput = Mat()
                Utils.bitmapToMat(originalBitmap, matInput)

                imageView.setImageBitmap(drawingBitmap)

                setupTouchListener()
                setupButtons()
            } finally {
                retriever.release()
            }
        }
    }

    private fun setupTouchListener() {
        imageView.setOnTouchListener { _, event ->
            if (!isROICropped) {  // ROI가 잘리지 않은 상태에서만 터치 이벤트 처리
                when (event.action) {
                    MotionEvent.ACTION_DOWN -> {
                        startX = event.x
                        startY = event.y
                        currentRect = null
                        cropButton.isEnabled = false
                        true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        drawRect(event.x, event.y)
                        true
                    }
                    MotionEvent.ACTION_UP -> {
                        drawRect(event.x, event.y)
                        cropButton.isEnabled = true
                        true
                    }
                    else -> false
                }
            }
            true
        }
    }

    private fun drawRect(endX: Float, endY: Float) {
        drawingBitmap = originalBitmap?.copy(Bitmap.Config.ARGB_8888, true)
        canvas = drawingBitmap?.let { Canvas(it) }

        val left = minOf(startX, endX)
        val top = minOf(startY, endY)
        val right = maxOf(startX, endX)
        val bottom = maxOf(startY, endY)

        currentRect = RectF(left, top, right, bottom)
        currentRect?.let { rect ->
            canvas?.drawRect(rect, paint)
        }

        imageView.setImageBitmap(drawingBitmap)
    }

    private fun setupButtons() {
        cropButton.setOnClickListener {
            currentRect?.let { rectF ->
                val rect = Rect(
                    rectF.left.toInt(),
                    rectF.top.toInt(),
                    rectF.right.toInt(),
                    rectF.bottom.toInt()
                )
                cropImage(rect)
            }
        }

        resetButton.setOnClickListener {
            // 초기 상태로 되돌리기
            isROICropped = false
            cropButton.visibility = View.VISIBLE
            resetButton.visibility = View.GONE
            cropButton.isEnabled = false
            currentRect = null

            // 원본 이미지 다시 표시
            drawingBitmap = originalBitmap?.copy(Bitmap.Config.ARGB_8888, true)
            canvas = drawingBitmap?.let { Canvas(it) }
            imageView.setImageBitmap(drawingBitmap)
        }

        finishROIButton.setOnClickListener {
            currentRect?.let { rectF ->
                // ImageView의 실제 표시 영역 가져오기
                val imageViewRect = RectF()
                imageView.getDrawingRect(Rect().apply {
                    imageView.getGlobalVisibleRect(this)
                    imageViewRect.set(this)
                })

                // 이미지의 실제 크기
                val imageWidth = originalBitmap?.width ?: 0
                val imageHeight = originalBitmap?.height ?: 0

                // 좌표 변환
                val scaleX = imageWidth / imageViewRect.width()
                val scaleY = imageHeight / imageViewRect.height()

                // ROI 좌표 저장 (스케일 적용)
                val roiData = ROIData(
                    (rectF.left * scaleX).toInt(),
                    (rectF.top * scaleY).toInt(),
                    (rectF.right * scaleX).toInt(),
                    (rectF.bottom * scaleY).toInt()
                )

                val intent = Intent(this, HSVActivity::class.java).apply {
                    putExtra("roiData", roiData)
                    putExtra("videoUri", getIntent().getStringExtra("videoUri"))
                }
                startActivity(intent)
                finish()
            }
        }
    }

    private fun cropImage(rect: Rect) {
        try {
            val width = rect.width()
            val height = rect.height()

            if (width <= 0 || height <= 0) {
                Toast.makeText(this, "유효한 영역을 선택해주세요", Toast.LENGTH_SHORT).show()
                return
            }

            originalBitmap?.let { bitmap ->
                val croppedBitmap = Bitmap.createBitmap(
                    bitmap,
                    rect.left,
                    rect.top,
                    width,
                    height
                )
                imageView.setImageBitmap(croppedBitmap)
                Toast.makeText(this, "ROI 영역이 잘렸습니다", Toast.LENGTH_SHORT).show()

                // ROI 잘린 후 상태 변경
                isROICropped = true
                cropButton.visibility = View.GONE
                resetButton.visibility = View.VISIBLE
            }
        } catch (e: Exception) {
            e.printStackTrace()
            Toast.makeText(this, "유효한 영역을 선택해주세요", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        matInput?.release()
    }
}

// ROI 데이터를 전달하기 위한 데이터 클래스
data class ROIData(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int
) : Parcelable {
    constructor(parcel: Parcel) : this(
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt()
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeInt(left)
        parcel.writeInt(top)
        parcel.writeInt(right)
        parcel.writeInt(bottom)
    }

    override fun describeContents(): Int = 0

    companion object CREATOR : Parcelable.Creator<ROIData> {
        override fun createFromParcel(parcel: Parcel): ROIData = ROIData(parcel)
        override fun newArray(size: Int): Array<ROIData?> = arrayOfNulls(size)
    }
}
