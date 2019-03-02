#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <cxcore.h>
#include <cv.h>


using namespace std;
using namespace cv;

#define PosSamNO 46    //正样本个数
#define NegSamNO 72    //负样本个数

#define TRAIN true    //是否进行训练,true表示重新训练，false表示读取xml文件中的SVM模型
#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体

//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 0


//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
    //获得SVM的决策函数中的alpha数组
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }
    
    //获得SVM的决策函数中的rho参数,即偏移量
    float get_rho()
    {
        return this->decision_func->rho;
    }
};



int main()
{
    //检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
    /*第一处*/HOGDescriptor hog(Size(64,64),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
    int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
    MySVM svm;//SVM分类器
    
    //若TRAIN为true，重新训练分类器
    if(TRAIN)
    {
        string ImgName;//图片名(绝对路径)
        ifstream finPos("/Users/ymy/Desktop/SVM2/SVM2/pos/pos.txt");//正样本图片的文件名列表
        ifstream finNeg("/Users/ymy/Desktop/SVM2/SVM2/neg/neg.txt");//负样本图片的文件名列表
        
        Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
        Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人
        
        
        //依次读取正样本图片，生成HOG描述子
        for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
        {
            cout<<"处理："<<ImgName<<endl;
            Mat src = imread(ImgName);//读取图片
            /*第二处*/resize(src,src,Size(64,64));
            //resize(src,src,Size(64,64));

            vector<float> descriptors;//HOG描述子向量
            //cout<< grad.cols;
            
            hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
            //cout<<"描述子维数："<<descriptors.size()<<endl;
            
            //处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
            if( 0 == num )
            {
                DescriptorDim = descriptors.size();//HOG描述子的维数
                //初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
                sampleFeatureMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, DescriptorDim, CV_32FC1);
                //初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
                sampleLabelMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, 1, CV_32FC1);
            }
            
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int i=0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
            sampleLabelMat.at<float>(num,0) = 1;//正样本类别为1，有人
        }
        
        //依次读取负样本图片，生成HOG描述子
        for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
        {
            cout<<"处理："<<ImgName<<endl;
            //cout<<ImgName<<endl;
            //waitKey();
            Mat src = imread(ImgName);//读取图片
            
             /*第三处*/resize(src,src,Size(128,128));
            
            vector<float> descriptors;//HOG描述子向量
            hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
            //cout<<"描述子维数："<<descriptors.size()<<endl;
            
            //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
            for(int i=0; i<DescriptorDim; i++)
                sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
            sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//负样本类别为-1，无人
        }
        
        //处理HardExample负样本
        if(HardExampleNO > 0)
        {
            ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本的文件名列表
            //依次读取HardExample负样本图片，生成HOG描述子
            for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
            {
                cout<<"处理："<<ImgName<<endl;
                ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
                Mat src = imread(ImgName);//读取图片
                //resize(src,img,Size(64,128));
                
                vector<float> descriptors;//HOG描述子向量
                hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
                //cout<<"描述子维数："<<descriptors.size()<<endl;
                
                //将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
                for(int i=0; i<DescriptorDim; i++)
                    sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];//第PosSamNO+num个样本的特征向量中的第i个元素
                sampleLabelMat.at<float>(num+PosSamNO+NegSamNO,0) = -1;//负样本类别为-1，无人
            }
        }
        

        //训练SVM分类器
        //迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
        CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
        //SVM参数：SVM类型为C_SVC；线性核函数；松弛因子C=0.01
        CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
        cout<<"开始训练SVM分类器"<<endl;
        svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//训练分类器
        cout<<"训练完成"<<endl;
        svm.save("/Users/ymy/Desktop/SVM_HOG.xml");//将训练好的SVM模型保存为xml文件
        
    }
    else //若TRAIN为false，从XML文件读取训练好的分类器
    {
        svm.load("/Users/ymy/Desktop/SVM_HOG.xml");//从XML文件读取训练好的SVM模型
    }
    
    
    /*************************************************************************************************
     线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
     将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
     如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
     就可以利用你的训练样本训练出来的分类器进行行人检测了。
     ***************************************************************************************************/
    DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
    int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
    cout<<"支持向量个数："<<supportVectorNum<<endl;
    
    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果
    
    //将支持向量的数据复制到supportVectorMat矩阵中
    for(int i=0; i<supportVectorNum; i++)
    {
        const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
        for(int j=0; j<DescriptorDim; j++)
        {
            //cout<<pData[j]<<" ";
            supportVectorMat.at<float>(i,j) = pSVData[j];
        }
    }
    
    //将alpha向量的数据复制到alphaMat中
    double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
    for(int i=0; i<supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }
    
    //计算-(alphaMat * supportVectorMat),结果放到resultMat中
    //gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
    resultMat = -1 * alphaMat * supportVectorMat;
    
    //得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
    vector<float> myDetector;
    //将resultMat中的数据复制到数组myDetector中
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }
    //最后添加偏移量rho，得到检测子
    myDetector.push_back(svm.get_rho());
    cout<<"检测子维数："<<myDetector.size()<<endl;
    //waitKey();
    //设置HOGDescriptor的检测子
     /*第四处*/HOGDescriptor myHOG(Size(64,64),Size(16,16),Size(8,8),Size(8,8),9);
    cout<<myHOG.getDescriptorSize()<<endl;
    //waitKey();
    myHOG.setSVMDetector(myDetector);
    //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    //保存检测子参数到文件
    ofstream fout("HOGDetectorForOpenCV.txt");
    for(int i=0; i<myDetector.size(); i++)
    {
        fout<<myDetector[i]<<endl;
    }
    
    /**************读入视频进行HOG行人检测******************/
    VideoCapture cap("/Users/ymy/Downloads/IMG_0249.MOV");//汽车正面检测
    Mat src;
    for(;;)
    {
        cap >> src;
        cap >> src;
        cap >> src;
        cap >> src;
        if( !src.empty() )
        {
            resize(src,src,Size(200,372));
            
            vector<Rect> found, found_filtered;//矩形框数组
            //cout<<"进行多尺度HOG人体检测"<<endl;
            myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测
            
            cout<<"找到的矩形框个数："<<found.size()<<endl;
            
            //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
            for(int i=0; i < found.size(); i++)
            {
                Rect r = found[i];
                int j=0;
                for(; j < found.size(); j++)
                    if(j != i && (r & found[j]) == r)
                        break;
                if( j == found.size())
                    found_filtered.push_back(r);
            }
            
            //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
            for(int i=0; i<found_filtered.size(); i++)
            {
                Rect r = found_filtered[i];
                r.x += cvRound(r.width*0.1);
                r.width = cvRound(r.width*0.8);
                r.y += cvRound(r.height*0.07);
                r.height = cvRound(r.height*0.8);
                rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
            }
            imshow("src",src);

        }
        else
        { printf(" --(!) No captured frame -- Break!"); break; }
        int c = waitKey(1);
        if( (char)c == 'c' ) { break; }
    }
    /****************************************************/
    /**************读入图片进行HOG行人检测******************
    //Mat src = imread("00000.jpg");
    //Mat src = imread("2007_000423.jpg");/Users/ymy/Downloads/2012-03-07-08-48-19.jpg
    Mat src = imread("/Users/ymy/Desktop/SVM2/SVM2/pos/9.png");//"/Users/ymy/Pictures/FN2V63AD2J.com.tencent.ScreenCapture2/QQ20160411-1@2x.png");///Users/ymy/Desktop/test.jpg");
    //resize(src,src,Size(200,372));
    resize(src,src,Size(100,100));
    
    vector<Rect> found, found_filtered;//矩形框数组
    //cout<<"进行多尺度HOG人体检测"<<endl;
    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测
    cout<<"找到的矩形框个数："<<found.size()<<endl;
    
    //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
    for(int i=0; i < found.size(); i++)
    {
        Rect r = found[i];
        int j=0;
        for(; j < found.size(); j++)
            if(j != i && (r & found[j]) == r)
                break;
        if( j == found.size())
            found_filtered.push_back(r);
    }
    
    //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
    for(int i=0; i<found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
    }
    
    imwrite("/Users/ymy/Desktop/ImgProcessed.jpg",src);

    imshow("src",src);
    waitKey();//注意：imshow之后必须加waitKey，否则无法显示图像
    /******************************************************************************/
    /******************批量处理负样本*************************************************
    string ImgN,ImgN1;
    ifstream finNeg("/Users/ymy/Desktop/SVM2/SVM2/neg/neg.txt");//负样本图片的文件名列表
    ifstream finNeg1("/Users/ymy/Desktop/SVM2/SVM2/neg/neg1.txt");
    
    getline(finNeg,ImgN);
    
    for(int i=1;i<491;i++)
    {
        cout<<"第"<<i<<"幅"<<endl;
        Mat src = imread(ImgN);
        getline(finNeg,ImgN);
        //resize(src,src,Size(200,200));
        
        vector<Rect> found, found_filtered;//矩形框数组
        //cout<<"进行多尺度HOG人体检测"<<endl;
        myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//对图片进行多尺度行人检测
        cout<<"找到的矩形框个数："<<found.size()<<endl;
        
        //找出所有没有嵌套的矩形框r,并放入found_filtered中,如果有嵌套的话,则取外面最大的那个矩形框放入found_filtered中
        for(int i=0; i < found.size(); i++)
        {
            Rect r = found[i];
            int j=0;
            for(; j < found.size(); j++)
                if(j != i && (r & found[j]) == r)
                    break;
            if( j == found.size())
                found_filtered.push_back(r);
        }
        
        //画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
        for(int i=0; i<found_filtered.size(); i++)
        {
            getline(finNeg1,ImgN1);
            Rect r = found_filtered[i];
            r.x += cvRound(r.width*0.1);
            r.width = cvRound(r.width*0.8);
            r.y += cvRound(r.height*0.07);
            r.height = cvRound(r.height*0.8);
            int a,b,c,d;
            if(r.tl().x>=0) a=r.tl().x;
            else a=0;
            if(r.tl().y>=0) b=r.tl().y;
            else b=0;
            if(r.br().x>src.cols) c=src.cols;
            else c=r.br().x;
            if(r.br().y>src.rows) d=src.rows;
            else d=r.br().y;
            Rect myROI(a,b,c-a,-b+d);
            cout<<myROI;
            Mat croppedImage=src(myROI);
            imwrite(ImgN1,croppedImage);
            
        }

    }
    /*************************************************/
    /******************读入单个64*128的测试图并对其HOG描述子进行分类*********************/
    ////读取测试图片(64*128大小)，并计算其HOG描述子
    ////Mat testImg = imread("person014142.jpg");
    //Mat testImg = imread("noperson000026.jpg");
    //vector<float> descriptor;
    //hog.compute(testImg,descriptor,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
    //Mat testFeatureMat = Mat::zeros(1,3780,CV_32FC1);//测试样本的特征向量矩阵
    ////将计算好的HOG描述子复制到testFeatureMat矩阵中
    //for(int i=0; i<descriptor.size(); i++)
    //  testFeatureMat.at<float>(0,i) = descriptor[i];
    
    ////用训练好的SVM分类器对测试图片的特征向量进行分类
    //int result = svm.predict(testFeatureMat);//返回类标
    //cout<<"分类结果："<<result<<endl;
    
    
    
    system("pause");
}