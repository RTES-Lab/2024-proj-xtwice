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

class DisplacementActivity : AppCompatActivity() {
    private lateinit var videoUri: String
    private lateinit var roiData: ROIData
    private lateinit var hsvRange: HSVRange
    private lateinit var markerPoints: List<MarkerPoint>
    private var fps: Float = 30f
    private lateinit var progressBar: ProgressBar
    private lateinit var statusTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_displacement)

        progressBar = findViewById(R.id.progressBar)
        statusTextView = findViewById(R.id.statusTextView)

        // 데이터 받기
        videoUri = intent.getStringExtra("videoUri") ?: ""
        roiData = intent.getParcelableExtra("roiData")!!
        hsvRange = intent.getParcelableExtra("hsvRange")!!
        markerPoints = intent.getParcelableArrayListExtra("markerPoints")!!
        fps = intent.getFloatExtra("fps", 30f)

        Log.d("DisplacementActivity", """
            HSV 범위 값:
            H: ${hsvRange.hMin} ~ ${hsvRange.hMax}
            S: ${hsvRange.sMin} ~ ${hsvRange.sMax}
            V: ${hsvRange.vMin} ~ ${hsvRange.vMax}
        """.trimIndent())

        // 백그라운드에서 변위 측정 실행
        Thread {
            measureDisplacementAndSaveCSV()
        }.start()
    }

    private fun measureDisplacementAndSaveCSV() {
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
        val frame = Mat()
        val displacements = mutableListOf<Pair<Float, Float>>()
        
        // 비디오 정보 로깅
        val totalFrames = videoCapture.get(Videoio.CAP_PROP_FRAME_COUNT).toInt()
        val videoFps = videoCapture.get(Videoio.CAP_PROP_FPS)
        val duration = totalFrames / videoFps
        
        Log.d("DisplacementActivity", """
            비디오 정보:
            - 총 프레임 수: $totalFrames
            - FPS: $videoFps
            - 예상 재생 시간: $duration 초
            - 초기 마커 위치: (${markerPoints[0].x}, ${markerPoints[0].y})
            - ROI: (${roiData.left}, ${roiData.top}, ${roiData.right}, ${roiData.bottom})
        """.trimIndent())

        // 초기 마커 위치
        val initialX = markerPoints[0].x
        val initialY = markerPoints[0].y
        var currentFrame = 0

        while (videoCapture.read(frame)) {
            currentFrame++
            if (frame.empty()) {
                Log.e("DisplacementActivity", "빈 프레임을 읽었습니다: $currentFrame")
                continue
            }
            
            // 프레임 회전
            val rotatedFrame = Mat()
            Core.rotate(frame, rotatedFrame, Core.ROTATE_90_CLOCKWISE)
            
            // ROI 범위 검증 추가
            val frameHeight = rotatedFrame.rows()  // 1080
            val frameWidth = rotatedFrame.cols()   // 1920
            
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
            val roi = rotatedFrame.submat(safeTop, safeBottom, safeLeft, safeRight)
            
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
            rotatedFrame.release()

            // 진행 상황 업데트
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

        runOnUiThread {
            statusTextView.text = "CSV 파일 저장 중..."
        }

        if (displacements.isNotEmpty()) {
            saveDisplacementsToCSV(displacements)
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
                writer.append("Frame,Time(s),DisplacementX(px),DisplacementY(px)\n")
                displacements.forEachIndexed { index, (dx, dy) ->
                    val timeInSeconds = index / fps
                    writer.append("$index,$timeInSeconds,$dx,$dy\n")
                }
            }
            
            runOnUiThread {
                showShareButton(csvFile)
                Toast.makeText(this, "CSV 파일이 저장되었습니다: ${csvFile.absolutePath}", Toast.LENGTH_LONG).show()
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

    private fun isExternalStorageDocument(uri: Uri) =
        "com.android.externalstorage.documents" == uri.authority

    private fun isDownloadsDocument(uri: Uri) =
        "com.android.providers.downloads.documents" == uri.authority

    private fun isMediaDocument(uri: Uri) =
        "com.android.providers.media.documents" == uri.authority
}
