<h1>Searching book cover</h1>

해당 프로그램은 파이썬, opencv를 이용하여 책 표지를 찾아주는 프로그램입니다.

해당 코드는 주피터 노트북에서 작성했습니다.



<h2>
    프로그램 사용법 및 과정
</h2>

![image-20211221170848488](C:\Users\aria1\AppData\Roaming\Typora\typora-user-images\image-20211221170848488.png)

1. 프로그램 실행시 카메라로 책을 상자안에 놓고 찍습니다(space버튼 입력)

![image-20211221171255133](C:\Users\aria1\AppData\Roaming\Typora\typora-user-images\image-20211221171255133.png)

2. 그러면 프로그램 내에서 상자 안의 사진만 다시 추출하여 출력합니다. 이 후 FLANN을 이용하여 두 이미지를 매칭하고, 상대적으로 좋은 매칭점만 추출하여 이미 저장되어 있는 사진들과 비교한 정확도를 저장합니다.

![image-20211221171913713](C:\Users\aria1\AppData\Roaming\Typora\typora-user-images\image-20211221171913713.png)

3.  프로그램이 비슷하다고 생각하는 사진을 골라 정확도와 함께 출력합니다. y버튼을 누르면 해당 책(사진 이름을 기반으로 함)을 네이버로 검색한 결과가 나옵니다. 다른 키를 누를 경우 프로그램을 종료합니다.



<h2>
    문제점
</h2>

정확도가 생각보다 높지 않아 추후 개선이 필요합니다.

또한 사진 이름이 아닌 이미지 자체로 검색을 하면 더 편리할 것입니다.
