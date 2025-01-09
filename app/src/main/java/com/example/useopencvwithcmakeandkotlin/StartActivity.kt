package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.os.Bundle
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity

class StartActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        supportActionBar?.hide()
        setContentView(R.layout.activity_start)

        findViewById<Button>(R.id.startButton).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
        }
    }
}