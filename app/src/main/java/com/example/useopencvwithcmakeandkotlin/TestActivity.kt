package com.example.useopencvwithcmakeandkotlin

import android.content.Context
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Tensor
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader
import java.io.IOException
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter

class TestActivity : AppCompatActivity() {
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        resultTextView = findViewById(R.id.resultTextView)

        // CSV 파일을 비동기적으로 읽고 모델을 실행
        GlobalScope.launch(Dispatchers.IO) {
            val data = readCsvAndParseListFromAssets("dataANN.csv")
            val modelFiles = assets.list("") // assets 폴더 내 모든 파일을 나열합니다.

            val results = mutableListOf<String>() // CSV로 저장할 결과 리스트


            modelFiles?.forEach { modelFile ->
                if (modelFile.endsWith(".tflite")) {
                    Log.d("TestActivity", "Processing model: $modelFile")
                    val modelResults = evaluateTFliteModel(modelFile, data)

                    val acc = modelResults.first
                    val loss = modelResults.second
                    val acc_per_label = modelResults.second

                    // 결과 로그 출력
                    Log.d("TestActivity", """TFlite Model: $modelFile - Accuracy: $acc,
                        |acc_per_label: $acc_per_label
                    """.trimMargin())

                    // CSV 결과에 추가
                    results.add("$modelFile,$acc,$loss")
                }
            }


            // 결과를 메인 스레드에서 처리
            withContext(Dispatchers.Main) {
                resultTextView.text = "Model evaluation completed."
            }
        }
    }


    fun readCsvAndParseListFromAssets(csvFileName: String): List<Triple<String, String, List<Float>>> {
        val data = mutableListOf<Triple<String, String, List<Float>>>()

        try {
            val inputStream = assets.open(csvFileName)
            val reader = BufferedReader(InputStreamReader(inputStream))

            // 첫 번째 줄 (헤더)을 건너뛰기
            reader.readLine()

            reader.forEachLine { line ->
                val columns = line.split(",").map { it.trim() }
                if (columns.size >= 3) {
                    val faultType = columns[0]
                    val label = columns[1]
                    val listString = columns[2] // z 열의 데이터
                    var rms_tmp = columns[3]
                    val fused = columns[7]

                    val rms = listOf(rms_tmp.toFloatOrNull() ?: 0f)


                    val parsedList = parseStringToList(listString)
                    data.add(Triple(faultType, label, rms))
                }
            }

            reader.close()
        } catch (e: Exception) {
            Log.e("TestActivity", "Error reading CSV from assets: ${e.message}")
        }

        return data
    }

    fun parseStringToList(listString: String): List<Float> {
        return listString
            .removeSurrounding("[", "]") // 대괄호 제거
            .replace("\"", "") // 큰따옴표 제거
            .trim() // 앞뒤 공백 제거
            .split("\\s+".toRegex()) // 공백을 기준으로 나누기
            .mapNotNull { it.toFloatOrNull() } // 각 항목을 Float으로 변환하고 실패 시 null 처리
    }

    private fun runPTLModel(inputData: List<Float>, modelFilePath: String): FloatArray {
        try {
            // 1. 모델 로드
            val model = LiteModuleLoader.load(modelFilePath)

            // 2. 입력 데이터 검증
            if (inputData.size != 2048) {
                throw IllegalArgumentException("Input data size must be 2048, but got ${inputData.size}")
            }

            // 3. 텐서 변환 및 추론
            val inputTensor = Tensor.fromBlob(inputData.toFloatArray(), longArrayOf(1, 1, 2048))

            // 4. 추론 실행 및 결과 반환
            return model.forward(IValue.from(inputTensor)).toTensor().dataAsFloatArray

        } catch (e: Exception) {
            Log.e("PyTorch", "Error in runModel: ${e.message}", e)
            throw e
        }
    }

    private fun runTFliteModel(inputData: List<Float>, modelFilePath: String): FloatArray {
        try {
            // 1. 모델 로드
            val modelFile = File(modelFilePath)
            val tfliteInterpreter = Interpreter(modelFile)

            // 2. 입력 데이터 검증
            if (inputData.size != 2048) {
                throw IllegalArgumentException("Input data size must be 2048, but got ${inputData.size}")
            }

            // 3. 텐서 변환 및 추론
            val inputTensor = Array(1) { Array(1) { FloatArray(2048) } }
            inputTensor[0][0] = inputData.toFloatArray()

            val outputTensor = Array(1) { FloatArray(4) }

            Log.d("TFLite", "Running inference on model")
            // 5. 추론 실행
            tfliteInterpreter.run(inputTensor, outputTensor)

            // 6. 결과 반환
            return outputTensor[0]

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

    private fun evaluateTFliteModel(
        modelFile: String,
        data: List<Triple<String, String, List<Float>>>
    ): Pair<Float, Map<Int, Float>> {
        var correct = 0
        var total = 0

        var model: MyModel
        model = MyModel(this, modelFile)

        // 라벨 별 정확도 계산을 위한 데이터 구조
        val labelCorrectCounts = mutableMapOf<Int, Int>()
        val labelTotalCounts = mutableMapOf<Int, Int>()

        data.forEach { (_, label, zList) ->
            val inputTensor = Array(1) { Array(1) { FloatArray(1) } }
            inputTensor[0][0] = zList.toFloatArray()

            // 예측 실행
            val output = model.predict(inputTensor)

            val prediction = output

            // 정확도 계산
            val predictedLabel = prediction.indices.maxByOrNull { prediction[it] } ?: -1
            val trueLabel = label.toInt()

            // 라벨 별 데이터 갱신
            labelTotalCounts[trueLabel] = labelTotalCounts.getOrDefault(trueLabel, 0) + 1
            if (trueLabel == predictedLabel) {
                correct++
                labelCorrectCounts[trueLabel] = labelCorrectCounts.getOrDefault(trueLabel, 0) + 1
            }

            total++
        }


//        val averageLoss = totalLoss / total
        val accuracy = correct.toFloat() / total

        // 라벨 별 정확도 계산
        val labelAccuracies = labelTotalCounts.mapValues { (label, count) ->
            val correctCount = labelCorrectCounts.getOrDefault(label, 0)
            correctCount.toFloat() / count
        }

        return Pair(accuracy, labelAccuracies)
    }

    private fun evaluatePTLModel(
        modelFilePath: String,
        data: List<Triple<String, String, List<Float>>>
    ): Pair<Float, Map<Int, Float>> {
//        var totalLoss = 0.0f
        var correct = 0
        var total = 0

        // 라벨 별 정확도 계산을 위한 데이터 구조
        val labelCorrectCounts = mutableMapOf<Int, Int>()
        val labelTotalCounts = mutableMapOf<Int, Int>()

        data.forEach { (_, label, zList) ->
            val prediction = runPTLModel(zList, modelFilePath)

            // 정확도 계산
            val predictedLabel = prediction.indices.maxByOrNull { prediction[it] } ?: -1
            val trueLabel = label.toInt()

            // 라벨 별 데이터 갱신
            labelTotalCounts[trueLabel] = labelTotalCounts.getOrDefault(trueLabel, 0) + 1
            if (trueLabel == predictedLabel) {
                correct++
                labelCorrectCounts[trueLabel] = labelCorrectCounts.getOrDefault(trueLabel, 0) + 1
            }

            total++
        }


//        val averageLoss = totalLoss / total
        val accuracy = correct.toFloat() / total

        // 라벨 별 정확도 계산
        val labelAccuracies = labelTotalCounts.mapValues { (label, count) ->
            val correctCount = labelCorrectCounts.getOrDefault(label, 0)
            correctCount.toFloat() / count
        }

        return Pair(accuracy, labelAccuracies)
    }

}
