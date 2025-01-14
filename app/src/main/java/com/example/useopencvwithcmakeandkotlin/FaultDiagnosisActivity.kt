package com.example.useopencvwithcmakeandkotlin

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class FaultDiagnosisActivity : AppCompatActivity() {

    private lateinit var resultTextView: TextView
    private lateinit var progressBar: ProgressBar
    private lateinit var statusTextView: TextView
    private lateinit var finishButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        supportActionBar?.hide()
        setContentView(R.layout.activity_result)

        // TextView에 추론 결과를 표시
        resultTextView = findViewById(R.id.resultTextView)
        resultTextView.movementMethod = android.text.method.LinkMovementMethod.getInstance()
        progressBar = findViewById(R.id.progressBar)
        statusTextView = findViewById(R.id.statusTextView)
        finishButton = findViewById(R.id.finishButton)

        val csvUriString = intent.getStringExtra("csvFileUri")
        if (csvUriString == null) {
            Toast.makeText(this, "CSV 파일 경로가 없습니다", Toast.LENGTH_LONG).show()
            finish()
            return
        }

        val csvUri = Uri.parse(csvUriString)
        val displacements = readCSVData(csvUri)

        // 파일 이름 로그 출력
        val csvFileName = File(csvUri.path ?: "").name
        Log.d("FaultDiagnosisActivity", "CSV 파일 이름: $csvFileName")

        if (displacements.size == 2048) {
            try {
                val modelOutput = runModel(displacements)

                val expValues = modelOutput.map { Math.exp(it.toDouble()) } // Float -> Double 변환
                val sumExpValues = expValues.sum()
                val probabilities = expValues.map { (it / sumExpValues).toFloat() } // 결과를 다시 Float로 변환
                Log.d("FaultDiagnosisActivity", "모델 출력: ${probabilities.joinToString(", ") { "%.2f".format(it) }}")

                val maxLogitIndex = modelOutput.indices.maxByOrNull { modelOutput[it] } ?: -1
                val classes = arrayOf("B", "H", "IR", "OR")
                val predictedClass = if (maxLogitIndex != -1) classes[maxLogitIndex] else "Unknown"

                // 결과 표시 및 UI 업데이트
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    statusTextView.visibility = View.GONE
                    
                    // 결과 텍스트 표시
                    resultTextView.apply {
                        visibility = View.VISIBLE
                        val spannableString = SpannableStringBuilder()
                        spannableString.append("결함 진단 결과\n\n")
                        
                        // "예상 결함: " 텍스트 추가
                        spannableString.append("예상 결함:  ")
                        
                        // 결함 종류와 확률을 SpannableString으로 생성
                        val resultText = "${predictedClass}  ${String.format("%.1f", probabilities[maxLogitIndex] * 100)}%"
                        val resultSpan = SpannableString(resultText)
                        
                        // 결함 종류에 따라 색상 설정
                        val color = when (predictedClass) {
                            "H" -> Color.parseColor("#1976D2")  // 파란색
                            else -> Color.parseColor("#D32F2F")  // 빨간색 (IR, B, OR의 경우)
                        }
                        
                        // 전체 텍스트에 볼드체와 색상 적용
                        resultSpan.setSpan(
                            StyleSpan(Typeface.BOLD),
                            0,
                            resultText.length,
                            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                        )
                        resultSpan.setSpan(
                            ForegroundColorSpan(color),
                            0,
                            resultText.length,
                            Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                        )
                        
                        spannableString.append(resultSpan)
                        spannableString.append("\n\n다른 결함 확률\n")
                        
                        // 다른 결함 확률 표시
                        classes.forEachIndexed { index, className ->
                            if (index != maxLogitIndex) {
                                val probText = "${className}  ${String.format("%.1f", probabilities[index] * 100)}%\n"
                                val probSpan = SpannableString(probText)
                                probSpan.setSpan(
                                    StyleSpan(Typeface.BOLD),
                                    className.length + 2,  // 클래스 이름과 공백 이후부터
                                    probText.length - 1,   // 줄바꿈 문자 전까지
                                    Spannable.SPAN_EXCLUSIVE_EXCLUSIVE
                                )
                                spannableString.append(probSpan)
                            }
                        }
                        
                        // TextView에 적용
                        text = spannableString
                        setLineSpacing(0f, 1.5f)  // 줄 간격 조정
                    }
                    
                    // 완료 버튼 표시
                    finishButton.apply {
                        visibility = View.VISIBLE
                        setOnClickListener {
                            // StartActivity로 이동하면서 현재까지의 액티비티 스택을 모두 제거
                            val intent = Intent(this@FaultDiagnosisActivity, StartActivity::class.java)
                            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
                            startActivity(intent)
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e("FaultDiagnosisActivity", "모델 실행 중 오류: ${e.message}")
                runOnUiThread {
                    progressBar.visibility = View.GONE
                    statusTextView.text = "오류: 분석 실패"
                }
            }
        } else {
            runOnUiThread {
                progressBar.visibility = View.GONE
                statusTextView.text = "오류: 데이터 부족"
            }
        }
    }

    private fun readCSVData(uri: Uri): List<Float> {
        val data = mutableListOf<Float>()
        try {
            val inputStream: InputStream? = contentResolver.openInputStream(uri)
            inputStream?.let {
                val reader = BufferedReader(InputStreamReader(it))
                reader.lineSequence()
                    .dropWhile { line -> line.startsWith("#") || line.startsWith("Frame") } // 헤더 제거
                    .take(2048) // 최대 2048개의 데이터만 읽음
                    .forEach { line ->
                        val values = line.split(",")
                        if (values.size >= 4) {
                            val displacementZ = values[3].toFloatOrNull() ?: 0f
                            data.add(displacementZ)
                        }
                    }
                reader.close()
            }
        } catch (e: Exception) {
            Log.e("FaultDiagnosisActivity", "CSV 읽기 중 오류: ${e.message}")
            e.printStackTrace()
        }
        return data
    }

    private fun runModel(inputData: List<Float>): FloatArray {
        try {
            // 1. 파일 경로 확인
            val modelFilePath = assetFilePath(this, "Fwdcnn7.ptl")
            Log.d("PyTorch", "Model path: $modelFilePath")

            // 2. 파일 존재 확인
            if (!File(modelFilePath).exists()) {
                throw IllegalStateException("Model file not found at $modelFilePath")
            }

            // 3. 모델 로드
            val model = LiteModuleLoader.load(modelFilePath)
            Log.d("PyTorch", "Model loaded successfully")

            // 4. 입력 데이터 검증

            if (inputData.size != 2048) {
                throw IllegalArgumentException("Input data size must be 2048, but got ${inputData.size}")
            }

            // 5. 텐서 변환 및 추론
            val inputTensor = Tensor.fromBlob(
                inputData.toFloatArray(),
                longArrayOf(1, 1, 2048)
            )

            // 6. 추론 실행 및 결과 반환
            return model.forward(IValue.from(inputTensor)).toTensor().dataAsFloatArray

        } catch (e: Exception) {
            Log.e("PyTorch", "Error in runModel: ${e.message}", e)
            throw e
        }
    }

    @Throws(IOException::class)
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var bytesRead: Int
                    while ((`is`.read(buffer).also { bytesRead = it }) != -1) {
                        os.write(buffer, 0, bytesRead)
                    }
                }
            }
        }
        return file.absolutePath
    }
}
