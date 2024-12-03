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
    private lateinit var confirmButton: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_video_size)

        widthEditText = findViewById(R.id.widthEditText)
        heightEditText = findViewById(R.id.heightEditText)
        confirmButton = findViewById(R.id.confirmButton)

        val videoUri = intent.getStringExtra("videoUri")

        confirmButton.setOnClickListener {
            val width = widthEditText.text.toString().toIntOrNull()
            val height = heightEditText.text.toString().toIntOrNull()

            if (width != null && height != null && width > 0 && height > 0) {
                val intent = Intent(this, ROIActivity::class.java).apply {
                    putExtra("videoUri", videoUri)
                    putExtra("videoWidth", width)
                    putExtra("videoHeight", height)
                }
                startActivity(intent)
                finish()
            } else {
                Toast.makeText(this, "올바른 크기를 입력해주세요", Toast.LENGTH_SHORT).show()
            }
        }
    }
}