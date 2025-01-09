package com.example.useopencvwithcmakeandkotlin

import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.ln

class TestActivity: AppCompatActivity() {
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        resultTextView = findViewById(R.id.resultTextView)

        // prediction_list를 담을 곳
        val predictionList = mutableListOf<IntArray>()
        var val_loss = 0.0f

        // CSV 파일을 비동기적으로 읽고 모델을 실행
        GlobalScope.launch(Dispatchers.IO) {
            val data = readCsvAndParseListFromAssets("data.csv")
            var correct = 0

            // 각 행의 z 데이터를 runModel에 전달하고 prediction_list에 저장
            data.forEachIndexed { index, (_, label, zList) ->
                val prediction = runModel(zList)
                val loss = crossEntropyLoss(arrayOf(prediction), intArrayOf(label.toInt()))
                val_loss += loss

                val predictedLabel = prediction.indices.maxByOrNull { prediction[it] } ?: -1
                predictionList.add(intArrayOf(predictedLabel))

                // label과 prediction 비교
                val actualLabel = label.toIntOrNull()
                if (actualLabel == predictedLabel) {
                    correct++
                }

                Log.d("TestActivity", "Row $index - Actual: $actualLabel, Predicted: $predictedLabel")
                Log.d("TestActivity", "Loss: $loss")
            }

            val_loss /= (data.size)
            // 결과를 메인 스레드에서 처리
            withContext(Dispatchers.Main) {
                Log.d("TestActivity", "data sizw: ${data.size}")
                Log.d("TestActivity", "Total predictions: ${predictionList.size}")
                Log.d("TestActivity", "Predictions: ${predictionList.joinToString { it.joinToString(", ", "[", "]") }}")
                Log.d("TestActivity", "Loss: $val_loss")
                resultTextView.text = """
                    |Correct predictions: $correct
                    |Loss: $val_loss""".trimMargin()
            }

        }
    }

    fun crossEntropyLoss(logits: Array<FloatArray>, labels: IntArray): Float {
        return logits.mapIndexed { index, sampleLogits ->
            val label = labels[index]
            require(label in sampleLogits.indices) {
                "레이블($label)이 유효한 범위(0-${sampleLogits.size - 1})를 벗어났습니다."
            }

            // Softmax 계산
            val expValues = sampleLogits.map { exp((it).toDouble()).toFloat() }
            val sumExp = expValues.sum()

            // 해당 클래스의 probability
            val probability = expValues[label] / sumExp

            // Cross Entropy 계산 (-log(probability))
            -ln(probability.toDouble()).toFloat()
        }.average().toFloat()
    }




    // CSV 파일을 읽고 데이터를 반환
    fun readCsvAndParseListFromAssets(csvFileName: String): List<Triple<String, String, List<Float>>> {
        val data = mutableListOf<Triple<String, String, List<Float>>>()
        val labels = mutableListOf<IntArray>()

        try {
            val inputStream = assets.open(csvFileName)
            val reader = BufferedReader(InputStreamReader(inputStream))

            // 첫 번째 줄 (헤더)을 건너뛰기
            reader.readLine()

            reader.forEachLine { line ->
                val columns = line.split(",").map { it.trim() }

                // 각 줄에 두 개 이상의 열이 있어야 함
                if (columns.size >= 3) {
                    val faultType = columns[0]
                    val label = columns[1]
                    val listString = columns[2] // z 열의 데이터

                    // 문자열로 저장된 리스트를 다시 리스트로 변환
                    val parsedList = parseStringToList(listString)

                    data.add(Triple(faultType, label, parsedList))
                    labels.add(label.split(" ").map { it.toInt() }.toIntArray())
                }
            }

            reader.close()
        } catch (e: Exception) {
            Log.e("TestActivity", "Error reading CSV from assets: ${e.message}")
        }

        return data
    }

    // 문자열로 된 리스트를 다시 리스트로 변환하는 함수
    fun parseStringToList(listString: String): List<Float> {
        return listString
            .removeSurrounding("[", "]") // 대괄호 제거
            .replace("\"", "") // 큰따옴표 제거
            .trim() // 앞뒤 공백 제거
            .split("\\s+".toRegex()) // 공백을 기준으로 나누기
            .mapNotNull { it.toFloatOrNull() } // 각 항목을 Float으로 변환하고 실패 시 null 처리
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