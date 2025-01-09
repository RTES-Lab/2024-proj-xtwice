package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.example.useopencvwithcmakeandkotlin.R
import com.example.useopencvwithcmakeandkotlin.ROIActivity

class VideoSizeActivity : AppCompatActivity() {
    private lateinit var widthEditText: EditText
    private lateinit var heightEditText: EditText
    private lateinit var fpsEditText: EditText
    private lateinit var confirmButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        supportActionBar?.hide()
        setContentView(R.layout.activity_video_size)

        widthEditText = findViewById(R.id.widthEditText)
        heightEditText = findViewById(R.id.heightEditText)
        fpsEditText = findViewById(R.id.fpsEditText)
        confirmButton = findViewById(R.id.confirmButton)

        val videoUri = intent.getStringExtra("videoUri")

        // 해상도 버튼 설정
        findViewById<Button>(R.id.resolution1080p).setOnClickListener {
            widthEditText.setText("1080")
            heightEditText.setText("1920")
        }

        findViewById<Button>(R.id.resolution1080pLand).setOnClickListener {
            widthEditText.setText("1920")
            heightEditText.setText("1080")
        }

        findViewById<Button>(R.id.resolution720p).setOnClickListener {
            widthEditText.setText("720")
            heightEditText.setText("1280")
        }

        findViewById<Button>(R.id.resolution720pLand).setOnClickListener {
            widthEditText.setText("1280")
            heightEditText.setText("720")
        }

        // FPS 버튼 설정
        findViewById<Button>(R.id.fps120).setOnClickListener {
            fpsEditText.setText("120")
        }

        findViewById<Button>(R.id.fps240).setOnClickListener {
            fpsEditText.setText("240")
        }

        confirmButton.setOnClickListener {
            val width = widthEditText.text.toString().toIntOrNull()
            val height = heightEditText.text.toString().toIntOrNull()
            val fps = fpsEditText.text.toString().toFloatOrNull()

            if (width != null && height != null && fps != null && 
                width > 0 && height > 0 && fps > 0) {
                val intent = Intent(this, ROIActivity::class.java).apply {
                    putExtra("videoUri", videoUri)
                    putExtra("videoWidth", width)
                    putExtra("videoHeight", height)
                    putExtra("fps", fps)
                }
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, "올바른 값을 입력해주세요", Toast.LENGTH_SHORT).show()
            }
        }
    }
}