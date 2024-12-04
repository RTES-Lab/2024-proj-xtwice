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
                            Log.d("ROIActivity", """
                                ACTION_DOWN Event:
                                Raw touch - x: ${event.x}, y: ${event.y}
                                Transformed to - x: $startX, y: $startY
                            """.trimIndent())
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
        // 원본 터치 좌표 로깅
        Log.d("ROIActivity", """
            Touch Coordinates:
            Raw touch - x: ${event.x}, y: ${event.y}
            Raw touch (getRawX/Y) - x: ${event.rawX}, y: ${event.rawY}
        """.trimIndent())

        val matrix = Matrix()
        imageView.imageMatrix.invert(matrix)

        // 이미지뷰 매트릭스 정보 로깅
        val matrixValues = FloatArray(9)
        imageView.imageMatrix.getValues(matrixValues)
        Log.d("ROIActivity", """
            ImageView Matrix:
            Scale X: ${matrixValues[Matrix.MSCALE_X]}
            Scale Y: ${matrixValues[Matrix.MSCALE_Y]}
            Translate X: ${matrixValues[Matrix.MTRANS_X]}
            Translate Y: ${matrixValues[Matrix.MTRANS_Y]}
        """.trimIndent())

        val touchPoint = floatArrayOf(event.x, event.y)
        matrix.mapPoints(touchPoint)

        // 변환된 좌표 로깅
        Log.d("ROIActivity", """
            Transformed Coordinates:
            Before transform - x: ${event.x}, y: ${event.y}
            After transform - x: ${touchPoint[0]}, y: ${touchPoint[1]}
        """.trimIndent())

        return PointF(touchPoint[0], touchPoint[1])
    }

    // 이미지뷰 내의 실제 이미지 영역을 계산하는 함수 추가
    private fun getImageBounds(imageView: ImageView): RectF {
        val viewWidth = imageView.width.toFloat()  // 1080
        
        // 원본 비디오 비율 유지하면서 크기 계산
        val originalRatio = originalBitmap?.width?.toFloat()!! / originalBitmap?.height?.toFloat()!!  // 1920/1080
        val scaledHeight = viewWidth / originalRatio  // 1080 / (1920/1080) ≈ 607.5
        
        // 이미지가 화면 중앙에 오도록 top 위치 계산
        val topMargin = (imageView.height - scaledHeight) / 2
        
        Log.d("ROIActivity", """
            Scaling Calculation:
            View Width: $viewWidth
            Original Ratio: $originalRatio
            Scaled Height: $scaledHeight
            Top Margin: $topMargin
        """.trimIndent())

        return RectF(0f, topMargin, viewWidth, topMargin + scaledHeight)
    }

    private fun drawRect(endX: Float, endY: Float) {
        drawingBitmap = originalBitmap?.copy(Bitmap.Config.ARGB_8888, true)
        canvas = drawingBitmap?.let { Canvas(it) }

        val left = minOf(startX, endX)
        val top = minOf(startY, endY)
        val right = maxOf(startX, endX)
        val bottom = maxOf(startY, endY)

        // ROI 계산 과정 로깅
        Log.d("ROIActivity", """
            ROI Calculation:
            Start point - x: $startX, y: $startY
            End point - x: $endX, y: $endY
            Calculated ROI - left: $left, top: $top, right: $right, bottom: $bottom
            ImageView size - width: ${imageView.width}, height: ${imageView.height}
            Original bitmap size - width: ${originalBitmap?.width}, height: ${originalBitmap?.height}
        """.trimIndent())

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
            // 좌표를 정수로 변환
            val left = rect.left.toInt()
            val top = rect.top.toInt()
            val right = rect.right.toInt()
            val bottom = rect.bottom.toInt()

            // 디버깅을 위한 로그
            Log.d("ROIActivity", """
                ROI Coordinates:
                Original rect: $rect
                Converted ROI: Left=$left, Top=$top, Right=$right, Bottom=$bottom
                Original bitmap size: ${originalBitmap?.width} x ${originalBitmap?.height}
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
