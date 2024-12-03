package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.os.Parcel
import android.os.Parcelable
import android.view.MotionEvent
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.Utils
import org.opencv.core.Mat
import kotlinx.parcelize.Parcelize

class MarkerCenterActivity : AppCompatActivity() {
    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }

    private lateinit var imageView: ImageView
    private lateinit var roiSizeSeekBar: SeekBar
    private lateinit var roiSizeTextView: TextView
    private lateinit var confirmButton: Button
    private lateinit var undoButton: Button
    private var matInput: Mat? = null
    private var roiData: ROIData? = null
    private var originalBitmap: Bitmap? = null
    private var drawingBitmap: Bitmap? = null
    private var canvas: Canvas? = null
    
    // 마커 좌표 저장
    private val markerCoordinates = mutableListOf<MarkerPoint>()
    private var roiSize: Int = 5
    private var croppedBitmap: Bitmap? = null  // ROI로 잘린 비트맵 저장용

    private val paint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.GREEN
        strokeWidth = 5f
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_marker_center)

        imageView = findViewById(R.id.imageView)
        roiSizeSeekBar = findViewById(R.id.roiSizeSeekBar)
        roiSizeTextView = findViewById(R.id.roiSizeTextView)
        confirmButton = findViewById(R.id.confirmButton)
        undoButton = findViewById(R.id.undoButton)

        // 이전 Activity에서 데이터 받기
        @Suppress("DEPRECATION")
        roiData = intent.getParcelableExtra<ROIData>("roiData")
        val videoUri = Uri.parse(intent.getStringExtra("videoUri"))
        @Suppress("DEPRECATION")
        val hsvRange = intent.getParcelableExtra<HSVRange>("hsvRange")
        @Suppress("DEPRECATION")
        val fps = intent.getParcelableExtra<FPS>("fps")
        
        loadAndProcessImage(videoUri)
        setupSeekBar()
        setupTouchListener()
        setupConfirmButton()
        setupUndoButton()
    }

    private fun loadAndProcessImage(videoUri: Uri?) {
        videoUri?.let {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this, videoUri)
                originalBitmap = retriever.getFrameAtTime(0)

                roiData?.let { roi ->
                    // 잘린 비트맵을 별도로 저장
                    croppedBitmap = Bitmap.createBitmap(
                        originalBitmap!!,
                        roi.left,
                        roi.top,
                        roi.right - roi.left,
                        roi.bottom - roi.top
                    )

                    matInput = Mat()
                    Utils.bitmapToMat(croppedBitmap, matInput)
                    drawingBitmap = croppedBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
                    canvas = Canvas(drawingBitmap!!)
                    imageView.setImageBitmap(drawingBitmap)
                }
            } finally {
                retriever.release()
            }
        }
    }

    private fun setupSeekBar() {
        roiSizeSeekBar.max = 300
        roiSizeSeekBar.progress = 5
        updateRoiSizeText(5)

        roiSizeSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                roiSize = progress
                updateRoiSizeText(progress)
                redrawMarkers()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun updateRoiSizeText(size: Int) {
        roiSizeTextView.text = "ROI 크기: $size px"
    }

    private fun setupTouchListener() {
        imageView.setOnTouchListener { view, event ->
            when (event.action) {
                MotionEvent.ACTION_DOWN -> {
                    // 터치 좌표를 이미지 좌표로 변환
                    val point = convertTouchPointToImagePoint(event.x, event.y)
                    markerCoordinates.add(MarkerPoint(point.x.toInt(), point.y.toInt()))
                    redrawMarkers()
                    true
                }
                else -> false
            }
        }
    }

    private fun convertTouchPointToImagePoint(touchX: Float, touchY: Float): android.graphics.PointF {
        val imageView = imageView
        val drawable = imageView.drawable
        val imageWidth = drawable.intrinsicWidth
        val imageHeight = drawable.intrinsicHeight
        
        // 이미지뷰의 실제 크기
        val viewWidth = imageView.width - imageView.paddingLeft - imageView.paddingRight
        val viewHeight = imageView.height - imageView.paddingTop - imageView.paddingBottom
        
        // 이미지의 실제 크기와 보여지는 크기의 비율 계산
        val scale: Float
        var offsetX = 0f
        var offsetY = 0f
        
        if (imageWidth * viewHeight > viewWidth * imageHeight) {
            // 이미지가 뷰보다 더 넓은 경우
            scale = viewWidth.toFloat() / imageWidth
            offsetY = (viewHeight - imageHeight * scale) / 2
        } else {
            // 이미지가 뷰보다 더 높은 경우
            scale = viewHeight.toFloat() / imageHeight
            offsetX = (viewWidth - imageWidth * scale) / 2
        }
        
        // 터치 좌표를 이미지 좌표로 변환
        val imageX = (touchX - offsetX) / scale
        val imageY = (touchY - offsetY) / scale
        
        return android.graphics.PointF(
            imageX.coerceIn(0f, imageWidth.toFloat()),
            imageY.coerceIn(0f, imageHeight.toFloat())
        )
    }

    private fun redrawMarkers() {
        // 항상 잘린 비트맵을 기준으로 다시 그리기
        drawingBitmap = croppedBitmap!!.copy(Bitmap.Config.ARGB_8888, true)
        canvas = Canvas(drawingBitmap!!)

        markerCoordinates.forEach { point ->
            // 마커 점 그리기
            paint.style = Paint.Style.FILL
            paint.color = Color.GREEN
            canvas?.drawCircle(point.x.toFloat(), point.y.toFloat(), 5f, paint)

            // ROI 박스 그리기
            paint.style = Paint.Style.STROKE
            paint.color = Color.BLUE
            val left = point.x - roiSize / 2
            val top = point.y - roiSize / 2
            val right = point.x + roiSize / 2
            val bottom = point.y + roiSize / 2
            canvas?.drawRect(
                left.toFloat(),
                top.toFloat(),
                right.toFloat(),
                bottom.toFloat(),
                paint
            )
        }

        imageView.setImageBitmap(drawingBitmap)
    }

    private fun setupConfirmButton() {
        confirmButton.setOnClickListener {
            if (markerCoordinates.isEmpty()) {
                Toast.makeText(this, "마커를 하나 이상 선택해주세요", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // 다음 DisplacementActivity로 데이터 전달
            val intent = Intent(this, DisplacementActivity::class.java).apply {
                putExtra("videoUri", getIntent().getStringExtra("videoUri"))
                putExtra("roiData", getIntent().getParcelableExtra<ROIData>("roiData"))
                putExtra("hsvRange", getIntent().getParcelableExtra<HSVRange>("hsvRange"))
                putExtra("markerPoints", ArrayList(markerCoordinates))
                putExtra("fps", getIntent().getParcelableExtra<FPS>("fps")?.value ?: 30f)
            }
            startActivity(intent)
            finish()
        }
    }

    private fun setupUndoButton() {
        undoButton.setOnClickListener {
            if (markerCoordinates.isNotEmpty()) {
                markerCoordinates.removeAt(markerCoordinates.lastIndex)
                redrawMarkers()
                Toast.makeText(this, "마지막 마커가 제거되었습니다", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "제거할 마커가 없습니다", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        matInput?.release()
    }
}

// 마커 위치를 저장할 데이터 클래스
data class MarkerPoint(
    val x: Int,
    val y: Int
) : Parcelable {
    constructor(parcel: Parcel) : this(
        parcel.readInt(),
        parcel.readInt()
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeInt(x)
        parcel.writeInt(y)
    }

    override fun describeContents(): Int = 0

    companion object CREATOR : Parcelable.Creator<MarkerPoint> {
        override fun createFromParcel(parcel: Parcel): MarkerPoint = MarkerPoint(parcel)
        override fun newArray(size: Int): Array<MarkerPoint?> = arrayOfNulls(size)
    }
}

// FPS 클래스 정의
data class FPS(val value: Float) : Parcelable {
    constructor(parcel: Parcel) : this(
        parcel.readFloat()
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeFloat(value)
    }

    override fun describeContents(): Int = 0

    companion object CREATOR : Parcelable.Creator<FPS> {
        override fun createFromParcel(parcel: Parcel): FPS = FPS(parcel)
        override fun newArray(size: Int): Array<FPS?> = arrayOfNulls(size)
    }
}

