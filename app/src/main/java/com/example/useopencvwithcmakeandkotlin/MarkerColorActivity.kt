package com.example.useopencvwithcmakeandkotlin
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.example.useopencvwithcmakeandkotlin.databinding.ActivityMarkerColorBinding
import android.util.Log






class MarkerColorActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMarkerColorBinding

    // 마커 색상별 HSV 범위 정의
    companion object {
        val COLOR_RANGES = mapOf(
            "BLUE" to HSVRange(
                hMin = 100, hMax = 140,
                sMin = 150, sMax = 255,
                vMin = 50, vMax = 255
            ),
            "GREEN" to HSVRange(
                hMin = 35, hMax = 85,
                sMin = 150, sMax = 255,
                vMin = 50, vMax = 255
            ),
            "WHITE" to HSVRange(
                hMin = 0, hMax = 180,
                sMin = 0, sMax = 30,
                vMin = 200, vMax = 255
            ),
            "YELLOW" to HSVRange(
                hMin = 20, hMax = 35,
                sMin = 150, sMax = 255,
                vMin = 50, vMax = 255
            ),
            "RED" to HSVRange(  // 빨간색은 HSV 색상환에서 양 끝에 위치
                hMin = 160, hMax = 180,
                sMin = 150, sMax = 255,
                vMin = 50, vMax = 255
            )
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMarkerColorBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupColorButtons()
    }

    private fun setupColorButtons() {
        binding.apply {
            blueButton.setOnClickListener { selectColor("BLUE") }
            greenButton.setOnClickListener { selectColor("GREEN") }
            whiteButton.setOnClickListener { selectColor("WHITE") }
            yellowButton.setOnClickListener { selectColor("YELLOW") }
            redButton.setOnClickListener { selectColor("RED") }
        }
    }

    private fun selectColor(color: String) {
        val hsvRange = COLOR_RANGES[color] ?: return
        
        val intent = Intent(this, HSVActivity::class.java).apply {
            putExtra("videoUri", getIntent().getStringExtra("videoUri"))
            putExtra("roiData", getIntent().getParcelableExtra<ROIData>("roiData"))
            putExtra("fps", getIntent().getFloatExtra("fps", 30f))
            putExtra("defaultHSVRange", hsvRange)
        }
        startActivity(intent)
        finish()
    }
} 