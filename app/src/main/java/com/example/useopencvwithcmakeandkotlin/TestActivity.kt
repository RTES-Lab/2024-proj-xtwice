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
import java.io.FileWriter
import androidx.appcompat.app.AppCompatActivity
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.exp
import kotlin.math.ln

class TestActivity : AppCompatActivity() {
    private lateinit var resultTextView: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_test)

        resultTextView = findViewById(R.id.resultTextView)

        // CSV 파일을 비동기적으로 읽고 모델을 실행
        GlobalScope.launch(Dispatchers.IO) {
            val data = readCsvAndParseListFromAssets("data.csv")
            val modelFiles = assets.list("") // assets 폴더 내 모든 파일을 나열합니다.

            val results = mutableListOf<String>() // CSV로 저장할 결과 리스트

            modelFiles?.forEach { modelFile ->
                if (modelFile.endsWith(".ptl")) {
                    Log.d("TestActivity", "Processing model: $modelFile")
                    val modelFilePath = assetFilePath(this@TestActivity, modelFile)
                    val modelResults = evaluateModel(modelFilePath, data)

                    val acc = modelResults.first
                    val loss = modelResults.second

                    // 결과 로그 출력
                    Log.d("TestActivity", "Model: $modelFile - Accuracy: $acc, Loss: $loss")

                    // CSV 결과에 추가
                    results.add("$modelFile,$acc,$loss")
                }
            }

            // CSV 파일로 저장
            writeResultsToCsv(results)

            // 결과를 메인 스레드에서 처리
            withContext(Dispatchers.Main) {
                resultTextView.text = "Model evaluation completed."
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
                    val parsedList = parseStringToList(listString)
                    data.add(Triple(faultType, label, parsedList))
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

    private fun runModel(inputData: List<Float>, modelFilePath: String): FloatArray {
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

    private fun evaluateModel(modelFilePath: String, data: List<Triple<String, String, List<Float>>>): Pair<Float, Float> {
        var totalLoss = 0.0f
        var correct = 0
        var total = 0
        data.forEach { (_, label, zList) ->
            val prediction = runModel(zList, modelFilePath)

            // Cross Entropy Loss 계산
            val loss = crossEntropyLoss(arrayOf(prediction), intArrayOf(label.toInt()))
            totalLoss += loss

            // 정확도 계산
            val predictedLabel = prediction.indices.maxByOrNull { prediction[it] } ?: -1
            if (label.toInt() == predictedLabel) {
                correct++
            }

            total++
        }
        val averageLoss = totalLoss / total
        val accuracy = correct.toFloat() / total
        return Pair(accuracy, averageLoss)
    }

    private fun writeResultsToCsv(results: List<String>) {
        try {
            val file = File(applicationContext.filesDir, "model_evaluation_results.csv")
            val writer = FileWriter(file)
            writer.append("Model,Accuracy,Loss\n")
            results.forEach { writer.append(it).append("\n") }
            writer.flush()
            writer.close()
        } catch (e: IOException) {
            Log.e("TestActivity", "Error writing CSV file: ${e.message}")
        }
    }
}
