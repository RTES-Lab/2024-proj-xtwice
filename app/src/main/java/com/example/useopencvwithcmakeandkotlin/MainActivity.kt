package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.os.Bundle
import android.provider.MediaStore
import androidx.appcompat.app.AppCompatActivity
import com.example.useopencvwithcmakeandkotlin.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 카메라 버튼 클릭 리스너 설정
        binding.cameraButton.setOnClickListener {
            val cameraIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            if (cameraIntent.resolveActivity(packageManager) != null) {
                startActivity(cameraIntent)
            }
        }

        // 갤러리 버튼 클릭 리스너 설정
        binding.galleryButton.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, REQUEST_VIDEO_PICK)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_VIDEO_PICK && resultCode == RESULT_OK) {
            val videoUri = data?.data
            videoUri?.let {
                val intent = Intent(this, ThumbnailActivity::class.java).apply {
                    putExtra("videoUri", it.toString())
                }
                startActivity(intent)
            }
        }
    }

    companion object {
        private const val REQUEST_VIDEO_PICK = 1
    }
}
