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
import android.util.Log
import android.graphics.PointF
import android.graphics.Matrix

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
    private var videoWidth: Int = 0
    private var videoHeight: Int = 0

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

        // 비디오 크기 정보 받기
        videoWidth = intent.getIntExtra("videoWidth", 0)
        videoHeight = intent.getIntExtra("videoHeight", 0)

        setupImageView()
    }

    private fun setupImageView() {
        imageView.scaleType = ImageView.ScaleType.FIT_CENTER  // 이미지 스케일 타입 지정
    }

    private fun setupTouchListener() {
        imageView.setOnTouchListener { view, event ->
            if (!isROICropped) {
                val bitmapCoords = getBitmapCoordinates(view as ImageView, event)

                if (bitmapCoords.x >= 0 && bitmapCoords.x < (originalBitmap?.width ?: 0) &&
                    bitmapCoords.y >= 0 && bitmapCoords.y < (originalBitmap?.height ?: 0)) {

                    when (event.action) {
                        MotionEvent.ACTION_DOWN -> {
                            startX = bitmapCoords.x
                            startY = bitmapCoords.y
                            currentRect = null
                            cropButton.isEnabled = false
                            true
                        }
                        MotionEvent.ACTION_MOVE -> {
                            drawRect(bitmapCoords.x, bitmapCoords.y)
                            true
                        }
                        MotionEvent.ACTION_UP -> {
                            drawRect(bitmapCoords.x, bitmapCoords.y)
                            cropButton.isEnabled = true
                            true
                        }
                        else -> false
                    }
                    true
                } else false
            } else false
        }
    }

    private fun getBitmapCoordinates(imageView: ImageView, event: MotionEvent): PointF {
        val matrix = Matrix()
        imageView.imageMatrix.invert(matrix)

        val touchPoint = floatArrayOf(event.x, event.y)
        matrix.mapPoints(touchPoint)

        return PointF(touchPoint[0], touchPoint[1])
    }

    // 이미지뷰 내의 실제 이미지 영역을 계산하는 함수 추가
    private fun getImageBounds(imageView: ImageView): RectF {
        val drawable = imageView.drawable ?: return RectF()
        val matrix = imageView.imageMatrix
        
        val bounds = RectF(0f, 0f, 
            drawable.intrinsicWidth.toFloat(),
            drawable.intrinsicHeight.toFloat())
            
        matrix.mapRect(bounds)

        // 이미지뷰의 크기와 이미지의 크기를 비교하여 여백 계산
        val viewWidth = imageView.width.toFloat()
        val viewHeight = imageView.height.toFloat()
        val imageWidth = bounds.width()
        val imageHeight = bounds.height()

        val leftPadding = (viewWidth - imageWidth) / 2
        val topPadding = (viewHeight - imageHeight) / 2

        return RectF(leftPadding, topPadding, leftPadding + imageWidth, topPadding + imageHeight)
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
            finishROI()
        }
    }

    private fun cropImage(rect: Rect) {
        try {
            val width = rect.width()
            val height = rect.height()

            if (width <= 0 || height <= 0) {
                Toast.makeText(this, "유효한 영역을 선택하세요", Toast.LENGTH_SHORT).show()
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

    private fun finishROI() {
        currentRect?.let { rect ->
            // 이미지뷰의 실제 이미지 영역 계산
            val imageRect = getImageBounds(imageView)
            
            // 디버깅을 위한 상세 로그 추가
            Log.d("ROIActivity", """
                Debug Info:
                Original Image Size: ${originalBitmap?.width} x ${originalBitmap?.height}
                ImageView Size: ${imageView.width} x ${imageView.height}
                Image Display Rect: ${imageRect.left}, ${imageRect.top}, ${imageRect.right}, ${imageRect.bottom}
                Selected ROI (Raw): ${rect.left}, ${rect.top}, ${rect.right}, ${rect.bottom}
                Selected ROI relative to ImageRect:
                Left: ${rect.left - imageRect.left}
                Top: ${rect.top - imageRect.top}
                Right: ${rect.right - imageRect.left}
                Bottom: ${rect.bottom - imageRect.top}
            """.trimIndent())

            // ROI 좌표를 원본 이미지 좌표계로 변환
            val scaleX = originalBitmap?.width?.toFloat()!! / (imageRect.right - imageRect.left)
            val scaleY = originalBitmap?.height?.toFloat()!! / (imageRect.bottom - imageRect.top)

            // 이미지뷰 내에서의 상대적인 위치 계산
            val relativeLeft = rect.left - imageRect.left
            val relativeTop = rect.top - imageRect.top
            val relativeRight = rect.right - imageRect.left
            val relativeBottom = rect.bottom - imageRect.top

            // 원본 이미지 좌표로 변환
            val left = (relativeLeft * scaleX).toInt().coerceIn(0, originalBitmap?.width ?: 0)
            val top = (relativeTop * scaleY).toInt().coerceIn(0, originalBitmap?.height ?: 0)
            val right = (relativeRight * scaleX).toInt().coerceIn(0, originalBitmap?.width ?: 0)
            val bottom = (relativeBottom * scaleY).toInt().coerceIn(0, originalBitmap?.height ?: 0)

            // 변환된 좌표 로그
            Log.d("ROIActivity", """
                Transformed Coordinates:
                Scale factors: scaleX=$scaleX, scaleY=$scaleY
                Final ROI: Left=$left, Top=$top, Right=$right, Bottom=$bottom
            """.trimIndent())

            if (left < right && top < bottom) {
                val roiData = ROIData(left, top, right, bottom)
                val intent = Intent(this, HSVActivity::class.java).apply {
                    putExtra("roiData", roiData)
                    putExtra("videoUri", getIntent().getStringExtra("videoUri"))
                }
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, "유효한 ROI 영역을 선택해주세요", Toast.LENGTH_SHORT).show()
            }
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
