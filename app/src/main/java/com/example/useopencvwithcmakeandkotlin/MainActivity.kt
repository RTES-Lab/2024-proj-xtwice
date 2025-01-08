package com.example.useopencvwithcmakeandkotlin

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import androidx.appcompat.app.AppCompatActivity
import com.example.useopencvwithcmakeandkotlin.databinding.ActivityMainBinding
import android.widget.Toast

// class 이름: MainActivity, parent: AppCompatActivity()
class MainActivity : AppCompatActivity() {

    // val: 불변 변수, var: 가변 변수
    // lateinit은 지연초기화로 kotlin에서 변수를 선언할 때 반드시 초기화를 해주어야 하는데 해당 변수가 언제 초기화될지는 알 수 없지만 ,
    // 반드시 초기화가 되고 이후에 사용된다는 것이 보장될 경우 lateinit을 사용한다.
    // private 으로 선언된 변수는 외부에서 직접 접근할 수 없고, 클래스 내부의 다른 멤버 함수에서만 사용가능
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val testButton = findViewById<Button>(R.id.testButton)

        // inflate() : ActivityMainBinding 클래스의 메서드. Activity의 레이아웃 파일을 인플레이션(뷰 객체로 변환) 한다.
        // layoutInflater : Activity의 레이아웃 인플레이터(Inflater)를 가리키는 객체로,
        // inflate() 메서드를 통해 Activity의 레이아웃 파일을 인플레이션 할 수 있다.
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // 갤러리 버튼 클릭 리스너 설정
        binding.galleryButton.setOnClickListener {
            val galleryIntent = Intent(Intent.ACTION_PICK, MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(galleryIntent, REQUEST_VIDEO_PICK)
        }

        binding.testButton.setOnClickListener(){
            val nextIntent = Intent(this, TestActivity::class.java)
            startActivity(nextIntent)
        }
    }

    // 함수 오버라이딩 - override fun
    // 함수 재정의(Function Overriding)을 하기 위한 키워드
    // 부모 클래스(MainActivity 클래스의 부모 클래스는 AppCompatActivity 클래스)에 open 키워드로 정의되어 있는 함수를 가져와
    // 다시 재정의 하는 것(함수 바디를 재정의).
    // onActivityResult(int requestCode, int resultCode, Intent data)
    //
    // int requestCode : subActivity를 호출했던 startActivityForResult()의 두번째 인수값
    //
    // int resultCode : 호출된 액티비티에서 설정한 성공(RESULT_OK)/실패(RESULT_CANCEL) 값
    //
    // Intent data : 호출된 액티비티에서 저장한 값
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_VIDEO_PICK && resultCode == RESULT_OK) {
            val videoUri = data?.data
            videoUri?.let {
                // 파일 이름 가져오기
                val fileName = getFileName(it)
                // 토스트 메시지 표시
                Toast.makeText(this, "'${fileName}'을(를) 선택하였습니다.", Toast.LENGTH_SHORT).show()
                handleSelectedVideo(it)
            }
        }
    }

    // 파일 이름을 가져오는 함수 추가
    private fun getFileName(uri: Uri): String {
        var fileName = "알 수 없는 파일"
        val cursor = contentResolver.query(uri, null, null, null, null)
        cursor?.use {
            if (it.moveToFirst()) {
                val displayNameIndex = it.getColumnIndex(MediaStore.MediaColumns.DISPLAY_NAME)
                if (displayNameIndex != -1) {
                    fileName = it.getString(displayNameIndex)
                }
            }
        }
        return fileName
    }

    // 비디오 선택 후 처리하는 부분
    private fun handleSelectedVideo(uri: Uri) {
        val intent = Intent(this, VideoSizeActivity::class.java).apply {
            putExtra("videoUri", uri.toString())
        }
        startActivity(intent)
    }

    companion object {
        private const val REQUEST_VIDEO_PICK = 1
    }
}
