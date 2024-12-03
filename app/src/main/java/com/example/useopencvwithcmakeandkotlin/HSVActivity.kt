package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import android.media.MediaMetadataRetriever
import android.os.Parcel
import android.os.Parcelable
import android.util.Log
import android.widget.Toast

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

    private lateinit var hSeekBarMin: SeekBar
    private lateinit var hSeekBarMax: SeekBar
    private lateinit var sMinSeekBar: SeekBar
    private lateinit var sMaxSeekBar: SeekBar
    private lateinit var vMinSeekBar: SeekBar
    private lateinit var vMaxSeekBar: SeekBar
    private lateinit var confirmButton: Button

    private var hMin = 0
    private var hMax = 179
    private var sMin = 0
    private var sMax = 255
    private var vMin = 0
    private var vMax = 255

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_hsv)

        imageView = findViewById(R.id.imageView)
        hsvInfoTextView = findViewById(R.id.hsvInfoTextView)

        roiData = intent.getParcelableExtra("roiData")
        val videoUri = intent.getStringExtra("videoUri")?.let { Uri.parse(it) }

        // ROI 데이터 로그 출력
        roiData?.let {
            Log.d("HSVActivity", "Received ROI Data - Left: ${it.left}, Top: ${it.top}, Right: ${it.right}, Bottom: ${it.bottom}")
        } ?: Log.e("HSVActivity", "ROI Data is null")

        loadAndProcessImage(videoUri)
        initializeSeekBars()
        setupSeekBarListeners()
        setupConfirmButton()
    }

    private fun loadAndProcessImage(videoUri: Uri?) {
        videoUri?.let {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this, videoUri)
                originalBitmap = retriever.getFrameAtTime(0)
                
                // ROI 데이터 로그 출력
                roiData?.let { roi ->
                    Log.d("HSVActivity", """
                        Loading image with ROI:
                        Original bitmap size: ${originalBitmap?.width} x ${originalBitmap?.height}
                        ROI coordinates: Left=${roi.left}, Top=${roi.top}, Right=${roi.right}, Bottom=${roi.bottom}
                    """.trimIndent())

                    // ROI 유효성 검사
                    if (roi.left < roi.right && roi.top < roi.bottom &&
                        roi.right <= (originalBitmap?.width ?: 0) &&
                        roi.bottom <= (originalBitmap?.height ?: 0)) {
                        
                        val croppedBitmap = Bitmap.createBitmap(
                            originalBitmap!!,
                            roi.left,
                            roi.top,
                            roi.right - roi.left,
                            roi.bottom - roi.top
                        )
                        
                        matInput = Mat()
                        Utils.bitmapToMat(croppedBitmap, matInput)
                        imageView.setImageBitmap(croppedBitmap)
                    } else {
                        Log.e("HSVActivity", "Invalid ROI coordinates")
                        Toast.makeText(this, "잘못된 ROI 좌표입니다", Toast.LENGTH_SHORT).show()
                    }
                } ?: Log.e("HSVActivity", "ROI data is null")
                
            } catch (e: Exception) {
                Log.e("HSVActivity", "Error processing image: ${e.message}")
                e.printStackTrace()
            } finally {
                retriever.release()
            }
        }
    }

    private fun initializeSeekBars() {
        hSeekBarMin = findViewById(R.id.hSeekBarMin)
        hSeekBarMax = findViewById(R.id.hSeekBarMax)
        sMinSeekBar = findViewById(R.id.sMinSeekBar)
        sMaxSeekBar = findViewById(R.id.sMaxSeekBar)
        vMinSeekBar = findViewById(R.id.vMinSeekBar)
        vMaxSeekBar = findViewById(R.id.vMaxSeekBar)
        confirmButton = findViewById(R.id.confirmButton)

        // 초기값 설정
        hSeekBarMax.progress = 179
        sMaxSeekBar.progress = 255
        vMaxSeekBar.progress = 255
    }

    private fun setupSeekBarListeners() {
        hSeekBarMin.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                hMin = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        hSeekBarMax.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                hMax = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        sMinSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                sMin = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        sMaxSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                sMax = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        vMinSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                vMin = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })

        vMaxSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                vMax = progress
                updateImageWithHSVFilter()
            }
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun updateImageWithHSVFilter() {
        matInput?.let { mat ->
            val hsvMat = Mat()
            val resultMat = Mat()

            // Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_BGRHSV)
            Imgproc.cvtColor(mat, hsvMat, Imgproc.COLOR_RGB2HSV)

            val lowerBound = Scalar(hMin.toDouble(), sMin.toDouble(), vMin.toDouble())
            val upperBound = Scalar(hMax.toDouble(), sMax.toDouble(), vMax.toDouble())
            Core.inRange(hsvMat, lowerBound, upperBound, resultMat)

            // 원본 이미지에 마스크 적용
            val filteredMat = Mat()
            mat.copyTo(filteredMat, resultMat)

            // 결과를 화면에 표시
            val resultBitmap = Bitmap.createBitmap(
                filteredMat.cols(),
                filteredMat.rows(),
                Bitmap.Config.ARGB_8888
            )
            Utils.matToBitmap(filteredMat, resultBitmap)
            imageView.setImageBitmap(resultBitmap)

            // 메모리 해제
            hsvMat.release()
            resultMat.release()
            filteredMat.release()
        }
    }

    private fun setupConfirmButton() {
        confirmButton.setOnClickListener {
            // AlertDialog로 HSV 범위 값 표시
            androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("선택된 HSV 범위")
                .setMessage("""
                    H: $hMin - $hMax
                    S: $sMin - $sMax
                    V: $vMin - $vMax
                    
                    이 값으로 선택하시겠습니까?
                """.trimIndent())
                .setPositiveButton("확인") { _, _ ->
                    // MarkerCenterActivity로 전환하면서 데이터 전달
                    val intent = Intent(this, MarkerCenterActivity::class.java).apply {
                        // HSV 범위 전달
                        putExtra("hsvRange", HSVRange(hMin, hMax, sMin, sMax, vMin, vMax))
                        // ROI 데이터 전달
                        putExtra("roiData", roiData)
                        // 비디오 URI 전달
                        putExtra("videoUri", getIntent().getStringExtra("videoUri"))
                    }
                    startActivity(intent)
                    finish()
                }
                .setNegativeButton("취소", null)
                .show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        matInput?.release()
    }
}

// HSV 범위를 저장할 데이터 클래스
data class HSVRange(
    val hMin: Int,
    val hMax: Int,
    val sMin: Int,
    val sMax: Int,
    val vMin: Int,
    val vMax: Int
) : Parcelable {
    constructor(parcel: Parcel) : this(
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt(),
        parcel.readInt()
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeInt(hMin)
        parcel.writeInt(hMax)
        parcel.writeInt(sMin)
        parcel.writeInt(sMax)
        parcel.writeInt(vMin)
        parcel.writeInt(vMax)
    }

    override fun describeContents(): Int = 0

    companion object CREATOR : Parcelable.Creator<HSVRange> {
        override fun createFromParcel(parcel: Parcel): HSVRange = HSVRange(parcel)
        override fun newArray(size: Int): Array<HSVRange?> = arrayOfNulls(size)
    }
}