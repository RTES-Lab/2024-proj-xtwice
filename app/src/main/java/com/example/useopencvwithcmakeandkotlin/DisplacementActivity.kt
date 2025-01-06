package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.io.File
import java.io.FileWriter
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.DocumentsContract
import android.provider.MediaStore
import android.content.ContentUris
import android.graphics.Bitmap
import android.widget.ImageView
import org.opencv.android.Utils

class DisplacementActivity : AppCompatActivity() {
    companion object {
        const val DEFAULT_FPS = 30f
    }

    private lateinit var videoUri: String
    private lateinit var roiData: ROIData
    private lateinit var hsvRange: HSVRange
    private lateinit var markerPoints: List<MarkerPoint>
    private var fps: Float = 30f
    private lateinit var progressBar: ProgressBar
    private lateinit var statusTextView: TextView
    private lateinit var firstFrameImageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_displacement)

        progressBar = findViewById(R.id.progressBar)
        statusTextView = findViewById(R.id.statusTextView)
        firstFrameImageView = findViewById(R.id.firstFrameImageView)

        // 데이터 받기 - null 체크 추가
        videoUri = intent.getStringExtra("videoUri") ?: ""
        roiData = intent.getParcelableExtra("roiData") 
            ?: throw IllegalStateException("ROI 데이터가 없습니다")
        hsvRange = intent.getParcelableExtra("hsvRange") 
            ?: throw IllegalStateException("HSV 범위 데이터가 없습니다")
        markerPoints = intent.getParcelableArrayListExtra("markerPoints") 
            ?: throw IllegalStateException("마커 포인트 데이터가 없습니다")
        fps = intent.getFloatExtra("fps", DEFAULT_FPS)

        // 로그 추가
        Log.d("DisplacementActivity", """
            전달받은 데이터:
            - videoUri: $videoUri
            - roiData: $roiData
            - hsvRange: $hsvRange
            - markerPoints: $markerPoints
            - fps: $fps
        """.trimIndent())

        // 백그라운드 작업 시작 전에 데이터 유효성 검사
        if (videoUri.isEmpty()) {
            Toast.makeText(this, "비디오 URI가 없습니다", Toast.LENGTH_SHORT).show()
            finish()
            return
        }

        // 백그라운드에서 변위 측정 실행
        Thread {
            try {
                measureDisplacementAndSaveCSV()
            } catch (e: Exception) {
                Log.e("DisplacementActivity", "변위 측정 중 오류 발생", e)
                runOnUiThread {
                    Toast.makeText(this, "오류가 발생했습니다: ${e.message}", Toast.LENGTH_LONG).show()
                    finish()
                }
            }
        }.start()
    }

    private fun measureDisplacementAndSaveCSV() {
        val fps = intent.getFloatExtra("fps", DEFAULT_FPS)
        if (fps == DEFAULT_FPS) {
            runOnUiThread {
                Toast.makeText(this, "FPS 값을 찾을 수 없어 기본값(30)을 사용합니다", Toast.LENGTH_SHORT).show()
            }
        }

        // URI를 실제 파일 경로로 변환
        val realPath = getRealPathFromURI(Uri.parse(videoUri))
        Log.d("DisplacementActivity", """
            비디오 정보:
            - 원본 URI: $videoUri
            - 실제 경로: $realPath
        """.trimIndent())
        
        if (realPath == null) {
            Log.e("DisplacementActivity", "비디오 파일 경로를 찾을 수 없습니다")
            runOnUiThread {
                Toast.makeText(this, "비디오 파일 경로를 찾을 수 없습니다", Toast.LENGTH_LONG).show()
                statusTextView.text = "오류: 파일 경로를 찾을 수 없음"
            }
            return
        }

        val videoCapture = VideoCapture(realPath)
        
        // FPS 설정 추가
        videoCapture.set(Videoio.CAP_PROP_FPS, fps.toDouble())  // Float를 Double로 변환
        
        // 비디오 정보 로깅
        val totalFrames = videoCapture.get(Videoio.CAP_PROP_FRAME_COUNT).toInt()
        val videoFps = videoCapture.get(Videoio.CAP_PROP_FPS)
        val duration = totalFrames / fps // 원래는 videoFps
        
        Log.d("DisplacementActivity", """
            비디오 설정 정보:
            - 총 프레임 수: $totalFrames
            - 원본 FPS: ${videoCapture.get(Videoio.CAP_PROP_FPS)}
            - 설정된 FPS: $fps
            - 예상 재생 시간: $duration 초
        """.trimIndent())

        // FPS 설정이 제대로 적용되었는지 확인
        if (videoCapture.get(Videoio.CAP_PROP_FPS).toFloat() != fps) {  // Double을 Float로 변환하여 비교
            Log.w("DisplacementActivity", "FPS 설정이 적용되지 않았습니다")
        }

        val frame = Mat()
        val displacements = mutableListOf<Pair<Float, Float>>()
        
        // 초기 마커 위치
        val initialX = markerPoints[0].x
        val initialY = markerPoints[0].y
        var currentFrame = 0

        // 첫 프레임 읽기
        if (videoCapture.read(frame)) {
            // 프전 코드 제거
            val displayFrame = frame.clone()
            
            // BGR을 RGB로 변환
            Imgproc.cvtColor(displayFrame, displayFrame, Imgproc.COLOR_BGR2RGB)
            
            // ROI 영역 표시를 위한 프레임 복사
            val safeTop = roiData.top.coerceIn(0, frame.rows())
            val safeBottom = roiData.bottom.coerceIn(0, frame.rows())
            val safeLeft = roiData.left.coerceIn(0, frame.cols())
            val safeRight = roiData.right.coerceIn(0, frame.cols())
            
            // ROI 영역 표시 (빨간색 사각형)
            Imgproc.rectangle(
                displayFrame,
                org.opencv.core.Point(safeLeft.toDouble(), safeTop.toDouble()),
                org.opencv.core.Point(safeRight.toDouble(), safeBottom.toDouble()),
                Scalar(255.0, 0.0, 0.0),  // RGB 형식
                2
            )
            
            // 결과 이미지를 화면에 표시
            val resultBitmap = Bitmap.createBitmap(
                displayFrame.cols(),
                displayFrame.rows(),
                Bitmap.Config.ARGB_8888 
            )
            Utils.matToBitmap(displayFrame, resultBitmap)
            
            runOnUiThread {
                firstFrameImageView.setImageBitmap(resultBitmap)
            }
            
            // 메모리 해제
            displayFrame.release()
        }
        
        // 나머지 프레임 처리 계속...
        var frameReadCount = 0
        while (videoCapture.read(frame)) {
            frameReadCount++
            if (frameReadCount % 100 == 0) {  // 100프레임마다 로그 출력
                Log.d("DisplacementActivity", "현재까지 읽은 프레임 수: $frameReadCount")
            }
            
            if (frame.empty()) {
                Log.e("DisplacementActivity", "빈 프레임: $frameReadCount")
                continue
            }
            
            // ROI 범위 검증 추가
            val frameHeight = frame.rows()
            val frameWidth = frame.cols()
            
            Log.d("DisplacementActivity", """
                프레임 정보:
                - 원본 크기: ${frame.cols()}x${frame.rows()}
                - 회전 후 크기: ${frameWidth}x${frameHeight} 
                - ROI: (${roiData.left}, ${roiData.top}, ${roiData.right}, ${roiData.bottom})
            """.trimIndent())
            
            // ROI 범위가 프레임을 벗어나지 않도록 조정
            val safeTop = roiData.top.coerceIn(0, frameHeight)
            val safeBottom = roiData.bottom.coerceIn(0, frameHeight)
            val safeLeft = roiData.left.coerceIn(0, frameWidth)
            val safeRight = roiData.right.coerceIn(0, frameWidth)
            
            // 안전한 ROI 범위로 서브매트릭스 추출
            val roi = frame.submat(safeTop, safeBottom, safeLeft, safeRight)
            
            try {
                // HSV 변환
                val hsvMat = Mat()
                Imgproc.cvtColor(roi, hsvMat, Imgproc.COLOR_RGB2HSV)

                // HSV 채널 분리해서 각 채널의 값 범위 확인
                val channels = ArrayList<Mat>()
                Core.split(hsvMat, channels)

                val hueRange = Core.minMaxLoc(channels[0])
                val satRange = Core.minMaxLoc(channels[1])
                val valRange = Core.minMaxLoc(channels[2])

                Log.d("DisplacementActivity", """
                    현재 프레임의 HSV 채널 정보:
                    H: ${hueRange.minVal} ~ ${hueRange.maxVal}
                    S: ${satRange.minVal} ~ ${satRange.maxVal}
                    V: ${valRange.minVal} ~ ${valRange.maxVal}
                    
                    설정된 HSV 범위:
                    H: ${hsvRange.hMin} ~ ${hsvRange.hMax}
                    S: ${hsvRange.sMin} ~ ${hsvRange.sMax}
                    V: ${hsvRange.vMin} ~ ${hsvRange.vMax}
                """.trimIndent())

                // HSV 범위로 마커 필터링
                val mask = Mat()
                Core.inRange(
                    hsvMat,
                    Scalar(hsvRange.hMin.toDouble(), hsvRange.sMin.toDouble(), hsvRange.vMin.toDouble()),
                    Scalar(hsvRange.hMax.toDouble(), hsvRange.sMax.toDouble(), hsvRange.vMax.toDouble()),
                    mask
                )

                // 노이즈 제거를 위한 모폴로지 연산 추가
                val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
                Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, kernel)
                Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, kernel)

                // 마스크의 픽셀 수 로그 출력
                val nonZeroCount = Core.countNonZero(mask)
                Log.d("DisplacementActivity", "마스크의 non-zero 픽셀 수: $nonZeroCount")

                // 마커의 중심점 찾기
                val moments = Imgproc.moments(mask)
                if (moments.m00 != 0.0) {
                    val currentX = (moments.m10 / moments.m00).toFloat()
                    val currentY = (moments.m01 / moments.m00).toFloat()
                    
                    // 디버깅을 위한 로그 추가
                    Log.d("DisplacementActivity", """
                        마커 검출 정보:
                        - ROI 내 절대 위치: ($currentX, $currentY)
                        - moments.m00: ${moments.m00}
                        - moments.m10: ${moments.m10}
                        - moments.m01: ${moments.m01}
                    """.trimIndent())
                    
                    // 변위 계산 없이 현재 위치를 그대로 저장
                    displacements.add(Pair(currentX, currentY))
                } else {
                    Log.w("DisplacementActivity", "마커를 찾을 수 없습니다")
                    if (displacements.isNotEmpty()) {
                        displacements.add(displacements.last())
                    } else {
                        displacements.add(Pair(0f, 0f))
                    }
                }

                // 메모리 해제
                hsvMat.release()
                mask.release()
                kernel.release()
                channels.forEach { it.release() }

            } catch (e: Exception) {
                Log.e("DisplacementActivity", "프레임 처리 중 오류: ${e.message}")
                e.printStackTrace()
            }

            // 메모리 해제
            roi.release()

            // 진행 상황 업데이트
            val progress = (currentFrame.toFloat() / totalFrames * 100).toInt()
            runOnUiThread {
                progressBar.progress = currentFrame
                statusTextView.text = "변위 측정 중... ($progress%)"
            }
        }

        videoCapture.release()
        frame.release()

        // 변위 데이터 확인
        Log.d("DisplacementActivity", "총 측정된 변위 데이터 수: ${displacements.size}")

        val avgX = displacements.map { it.first }.average().toFloat()
        val avgY = displacements.map { it.second }.average().toFloat()
        Log.d("DisplacementActivity", "변위 평균값 계산됨: X = $avgX, Y = $avgY")

        val adjustedDisplacements = displacements.map {
            Pair(it.first - avgX, it.second - avgY)
        }


        runOnUiThread {
            statusTextView.text = "CSV 파일 저장 중..."
        }

        if (displacements.isNotEmpty()) {
            saveDisplacementsToCSV(adjustedDisplacements)
            runOnUiThread {
                statusTextView.text = "저장 완료! (${displacements.size}개 데이터)"
            }
        } else {
            Log.e("DisplacementActivity", "저장할 변위 데이터가 없습니다")
            runOnUiThread {
                statusTextView.text = "오류: 변위 데이터 없음"
                Toast.makeText(this, "변위 데이터를 측정하지 못했습니다", Toast.LENGTH_LONG).show()
            }
        }

        // 변위 추출 완료 후 요약 정보 로그 출력
        Log.d("DisplacementActivity", """
            변위 추출 완료:
            - 총 프레임 수: $totalFrames
            - 실제 FPS: $videoFps
            - 설정된 FPS: $fps
            - 측정된 변위 데이터 수: ${displacements.size}
        """.trimIndent())

        runOnUiThread {
            Toast.makeText(this, "변위 측정이 완료되었습니다.", Toast.LENGTH_SHORT).show()
        }

        Log.d("DisplacementActivity", """
            프레임 처리 결과:
            - 총 읽은 프레임 수: $frameReadCount
            - 저장된 변위 데이터 수: ${displacements.size}
        """.trimIndent())
    }

    private fun saveDisplacementsToCSV(displacements: List<Pair<Float, Float>>) {
        val publicDir = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOCUMENTS)
        val appFolder = File(publicDir, "OpenCVDisplacement")
        if (!appFolder.exists()) {
            appFolder.mkdirs()
        }

        val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val csvFile = File(appFolder, "displacement_${timeStamp}.csv")

        try {
            FileWriter(csvFile).use { writer ->
                // 헤더에 FPS 정보 추가
                writer.append("# FPS: $fps\n")
                writer.append("Frame,Time(s),DisplacementX(px),DisplacementZ(px)\n")
                displacements.forEachIndexed { index, (dx, dz) ->
                    val timeInSeconds = index / fps
                    writer.append("$index,$timeInSeconds,$dx,$dz\n")
                }
            }

            runOnUiThread {
                showShareButton(csvFile)
                Toast.makeText(this, "CSV 파일이 저장되었습니다: ${csvFile.absolutePath}", Toast.LENGTH_LONG).show()
                showInferenceButton(csvFile)
            }
        } catch (e: IOException) {
            e.printStackTrace()
            runOnUiThread {
                Toast.makeText(this, "CSV 파일 저장 실패", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun showShareButton(file: File) {
        val shareButton = findViewById<Button>(R.id.shareButton)
        shareButton.visibility = View.VISIBLE
        shareButton.setOnClickListener {
            val uri = FileProvider.getUriForFile(
                this,
                "${packageName}.provider",
                file
            )
            val shareIntent = Intent().apply {
                action = Intent.ACTION_SEND
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "CSV 파일 공유"))
        }
    }

    private fun getRealPathFromURI(uri: Uri): String? {
        try {
            when {
                // 미디어 저장소에서 가져온 경우
                DocumentsContract.isDocumentUri(this, uri) -> {
                    val docId = DocumentsContract.getDocumentId(uri)
                    when {
                        isExternalStorageDocument(uri) -> {
                            val split = docId.split(":")
                            val type = split[0]
                            if ("primary".equals(type, ignoreCase = true)) {
                                return "${Environment.getExternalStorageDirectory()}/${split[1]}"
                            }
                        }
                        isDownloadsDocument(uri) -> {
                            val contentUri = ContentUris.withAppendedId(
                                Uri.parse("content://downloads/public_downloads"),
                                docId.toLong()
                            )
                            return getDataColumn(this, contentUri, null, null)
                        }
                        isMediaDocument(uri) -> {
                            val split = docId.split(":")
                            val type = split[0]
                            var contentUri: Uri? = null
                            when (type) {
                                "image" -> contentUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
                                "video" -> contentUri = MediaStore.Video.Media.EXTERNAL_CONTENT_URI
                                "audio" -> contentUri = MediaStore.Audio.Media.EXTERNAL_CONTENT_URI
                            }
                            val selection = "_id=?"
                            val selectionArgs = arrayOf(split[1])
                            return getDataColumn(this, contentUri, selection, selectionArgs)
                        }
                    }
                }
                // 일반 미디어 파일인 경우
                "content".equals(uri.scheme, ignoreCase = true) -> {
                    return getDataColumn(this, uri, null, null)
                }
                // 파일 경로인 경우
                "file".equals(uri.scheme, ignoreCase = true) -> {
                    return uri.path
                }
            }
        } catch (e: Exception) {
            Log.e("DisplacementActivity", "getRealPathFromURI 오류: ${e.message}")
            e.printStackTrace()
        }
        return null
    }

    private fun getDataColumn(context: Context, uri: Uri?, selection: String?, selectionArgs: Array<String>?): String? {
        uri?.let {
            val projection = arrayOf(MediaStore.MediaColumns.DATA)
            try {
                context.contentResolver.query(it, projection, selection, selectionArgs, null)?.use { cursor ->
                    if (cursor.moveToFirst()) {
                        val columnIndex = cursor.getColumnIndexOrThrow(MediaStore.MediaColumns.DATA)
                        return cursor.getString(columnIndex)
                    }
                }
            } catch (e: Exception) {
                Log.e("DisplacementActivity", "getDataColumn 오류: ${e.message}")
                e.printStackTrace()
            }
        }
        return null
    }

    private fun showInferenceButton(file: File) {
        val inferenceButton = findViewById<Button>(R.id.inferenceButton)
        inferenceButton.visibility = View.VISIBLE
        inferenceButton.setOnClickListener {
            val intent = Intent(this, FaultDiagnosisActivity::class.java).apply {
                putExtra("csvFileUri", Uri.fromFile(file).toString())
            }
            startActivity(intent)
        }
    }


    private fun isExternalStorageDocument(uri: Uri) =
        "com.android.externalstorage.documents" == uri.authority

    private fun isDownloadsDocument(uri: Uri) =
        "com.android.providers.downloads.documents" == uri.authority

    private fun isMediaDocument(uri: Uri) =
        "com.android.providers.media.documents" == uri.authority
}