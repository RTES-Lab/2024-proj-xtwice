package com.example.modelapp.model

import android.content.Context
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.channels.FileChannel

class MyModel(context: Context) {
    private var interpreter: Interpreter

    init {
        val assetManager = context.assets
        val modelFile = assetManager.openFd("ANN_rms.tflite")
        val inputStream = FileInputStream(modelFile.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = modelFile.startOffset
        val declaredLength = modelFile.declaredLength

        val modelBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            startOffset,
            declaredLength
        )

        modelBuffer.order(ByteOrder.nativeOrder())

        interpreter = Interpreter(modelBuffer)

        // 리소스 닫기
        inputStream.close()
    }

    // 입력을 FloatArray로 받고 출력도 FloatArray로 반환
    fun predict(input: FloatArray): FloatArray {
        // 입력을 [1, 1] 형태로 변환
        val input2D = arrayOf(input)

        // 출력 데이터 [1, 4] 생성
        val output = Array(1) { FloatArray(4) }

        // 예측 실행
        interpreter.run(input2D, output)

        // 1차원 출력 반환
        return output[0]
    }

    // 사용이 끝나면 인터프리터를 닫는 메서드 추가
    fun close() {
        interpreter.close()
    }
}
