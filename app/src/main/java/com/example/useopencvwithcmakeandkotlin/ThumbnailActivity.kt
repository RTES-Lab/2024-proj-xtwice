package com.example.useopencvwithcmakeandkotlin

import android.graphics.Bitmap
import android.graphics.Matrix
import android.media.ThumbnailUtils
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.MotionEvent
import android.view.ScaleGestureDetector
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.RelativeLayout
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class ThumbnailActivity : AppCompatActivity() {

    private lateinit var imageView: ImageView
    private lateinit var selectionView: View
    private lateinit var cropButton: Button
    private var scaleGestureDetector: ScaleGestureDetector? = null
    private var matrix = Matrix()
    private var scaleFactor = 1.0f
    private var startX: Float = 0f
    private var startY: Float = 0f
    private var endX: Float = 0f
    private var endY: Float = 0f
    private var thumbnail: Bitmap? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_thumbnail)

        imageView = findViewById(R.id.imageView)
        selectionView = findViewById(R.id.selectionView)
        cropButton = findViewById(R.id.cropButton)

        val videoUri = intent.getStringExtra("videoUri")?.let { Uri.parse(it) }
        videoUri?.let {
            val videoPath = getRealPathFromURI(it)
            thumbnail = ThumbnailUtils.createVideoThumbnail(
                videoPath,
                MediaStore.Images.Thumbnails.MINI_KIND
            )
            imageView.setImageBitmap(thumbnail)
        }

        scaleGestureDetector = ScaleGestureDetector(this, ScaleListener())

        imageView.setOnTouchListener { _, event ->
            scaleGestureDetector?.onTouchEvent(event)
            handleTouch(event)
            true
        }

        cropButton.setOnClickListener {
            cropImage()
        }
    }

    private fun handleTouch(event: MotionEvent) {
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                startX = event.x
                startY = event.y
                selectionView.visibility = View.VISIBLE
            }
            MotionEvent.ACTION_MOVE -> {
                endX = event.x
                endY = event.y
                updateSelectionView()
            }
            MotionEvent.ACTION_UP -> {
                endX = event.x
                endY = event.y
                updateSelectionView()
            }
        }
    }

    private fun updateSelectionView() {
        val left = startX.coerceAtMost(endX).toInt()
        val right = startX.coerceAtLeast(endX).toInt()
        val top = startY.coerceAtMost(endY).toInt()
        val bottom = startY.coerceAtLeast(endY).toInt()

        val layoutParams = selectionView.layoutParams as RelativeLayout.LayoutParams
        layoutParams.leftMargin = left
        layoutParams.topMargin = top
        layoutParams.width = right - left
        layoutParams.height = bottom - top
        selectionView.layoutParams = layoutParams
    }

    private fun cropImage() {
        thumbnail?.let {
            // ROI 범위 계산
            val x = (startX / scaleFactor).toInt()
            val y = (startY / scaleFactor).toInt()
            val width = ((endX - startX) / scaleFactor).toInt()
            val height = ((endY - startY) / scaleFactor).toInt()

            // ROI 범위가 이미지 크기를 초과하는지 확인
            if (x < 0 || y < 0 || x + width > it.width || y + height > it.height) {
                // 범위를 넘으면 메시지 표시
                Toast.makeText(this, "선택한 영역이 이미지 밖으로 넘어갔습니다. 다시 선택해주세요.", Toast.LENGTH_SHORT).show()
            } else {
                // ROI 범위가 올바른 경우 이미지를 자르고 중앙에 표시
                val croppedBitmap = Bitmap.createBitmap(it, x, y, width, height)
                imageView.setImageBitmap(croppedBitmap)

                // 중앙에 맞추기 위해 Matrix 조정
                val viewWidth = imageView.width
                val viewHeight = imageView.height
                val bitmapWidth = croppedBitmap.width
                val bitmapHeight = croppedBitmap.height

                // 중심 맞추기 위한 비율 계산
                val offsetX = (viewWidth - bitmapWidth * scaleFactor) / 2
                val offsetY = (viewHeight - bitmapHeight * scaleFactor) / 2

                matrix.reset()
                matrix.postTranslate(offsetX, offsetY)
                matrix.postScale(scaleFactor, scaleFactor, viewWidth / 2f, viewHeight / 2f)
                imageView.imageMatrix = matrix

                selectionView.visibility = View.GONE
            }
        }
    }

    private fun getRealPathFromURI(contentUri: Uri): String {
        var result: String? = null
        val cursor = contentResolver.query(contentUri, null, null, null, null)
        if (cursor != null) {
            if (cursor.moveToFirst()) {
                val idx = cursor.getColumnIndex(MediaStore.Video.Media.DATA)
                result = cursor.getString(idx)
            }
            cursor.close()
        }
        if (result == null) {
            result = contentUri.path
        }
        return result ?: ""
    }

    private inner class ScaleListener : ScaleGestureDetector.SimpleOnScaleGestureListener() {
        override fun onScale(detector: ScaleGestureDetector): Boolean {
            scaleFactor *= detector.scaleFactor
            scaleFactor = scaleFactor.coerceIn(0.1f, 10.0f)
            matrix.setScale(scaleFactor, scaleFactor)
            imageView.imageMatrix = matrix
            return true
        }
    }
}
