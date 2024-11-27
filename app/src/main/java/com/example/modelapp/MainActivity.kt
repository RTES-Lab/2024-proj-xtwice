package com.example.modelapp

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.modelapp.model.MyModel

class MainActivity : AppCompatActivity() {
    private lateinit var model: MyModel

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        model = MyModel(this)

        val predictButton: Button = findViewById(R.id.predictButton)
        val resultText: TextView = findViewById(R.id.resultText)

        predictButton.setOnClickListener {
            // 입력값 설정 (FloatArray로 1개 값을 사용)
            val input = FloatArray(1) {  0.0208934f }

            // 예측 실행
            val output = model.predict(input)

            // 클래스별 확률 출력
            val probabilities = output.joinToString(separator = ", ") { "%.4f".format(it) }
            val maxProbabilityIndex = output.indices.maxByOrNull { output[it] } ?: -1
            val classes = arrayOf("B", "H", "IR", "OR")
            val predictedClass = if (maxProbabilityIndex != -1) classes[maxProbabilityIndex] else "Unknown"

            resultText.text = "Class Probabilities: [$probabilities]\nPredicted Class: $predictedClass"

        }
    }
}
