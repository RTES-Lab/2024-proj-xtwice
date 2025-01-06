package com.example.useopencvwithcmakeandkotlin

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Button
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

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_result)

        var runmodelbtn = findViewById<Button>(R.id.runModelButton)

        // TextView에 추론 결과를 표시
        resultTextView = findViewById(R.id.resultTextView)

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

        runmodelbtn.setOnClickListener {
            if (displacements.size == 2048) {
                val modelOutput = runModel(displacements)

                // 로짓 값 출력
                val probabilities = "Model Output: ${modelOutput.joinToString(", ")}"

                // 로짓 값에서 가장 큰 값을 가진 인덱스 찾기
                val maxLogitIndex = modelOutput.indices.maxByOrNull { modelOutput[it] } ?: -1
                val classes = arrayOf("B", "H", "IR", "OR")
                val predictedClass =
                    if (maxLogitIndex != -1) classes[maxLogitIndex] else "Unknown"

                resultTextView.text =
                    "Class Logits: \n[$probabilities]\nPredicted Class: $predictedClass"
            } else {
                Toast.makeText(this, "CSV 파일 데이터가 부족합니다", Toast.LENGTH_LONG).show()
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
                        println(values)
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
            val modelFilePath = assetFilePath(this, "wdcnn.ptl")
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
