package com.example.cvlab.androidcamera;



import android.content.Context;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.CornerPathEffect;
import android.graphics.DiscretePathEffect;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.YuvImage;
import android.hardware.Camera;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.Toast;

import com.google.android.gms.appindexing.Action;
import com.google.android.gms.appindexing.AppIndex;
import com.google.android.gms.appindexing.Thing;
import com.google.android.gms.common.api.GoogleApiClient;
import com.sh1r0.caffe_android_lib.CaffeMobile;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import static android.R.attr.delay;
import static java.lang.Math.pow;
import static org.opencv.core.Core.flip;
import static org.opencv.core.Core.transpose;


/**
 * Created by Dell on 2016/7/12.
 */
public class MainActivity extends AppCompatActivity implements Camera.PreviewCallback {


    static {
        if (!OpenCVLoader.initDebug()) {
            System.out.println("opencv 初始化失败！");
        } else {
            System.loadLibrary("opencv_java3");

        }
    }

    private static final String LOG_TAG = "MainActivity";

    SurfaceView mSurfaceview1, mSurfaceview2; // SurfaceView对象：(视图组件)视频显示
    SurfaceHolder mSurfaceHolder1, mSurfaceHolder2; // SurfaceHolder对象：(抽象接口)SurfaceView支持类
    Camera mCamera; // Camera对象，相机预览

    private Mat yuvImg,yuvImg2;
    private Mat grayscaleImage;

    int screenWidth, screenHeight, previewWidth, previewHeight;
    public Mat rotateMat = null;
    boolean isPreview = false;
    private Thread drawThread;
    public boolean isRun = false;
    public String FPS;
    public boolean points_recognized = false;
    float X_SCALE, Y_SCALE;
    private int absolutePointSize;
    private CascadeClassifier cascadeClassifier;
    public Date lastTime = new Date(System.currentTimeMillis());
    public Mat rotateImg = null;
    // public org.opencv.core.Rect[] pointsArray;
    public Rect pointsArray;
    private int num = 0;
    public float sizeX, sizeY;
    List<Rect> pointarray = new ArrayList<>();
    public double Distance;
    public Rect newfaceArray = new Rect();
    public Rect newpointArray = new Rect();
    public int x1, x2, x3, y1, y2, y3;
    public int w1, w2, w3, z1, z2, z3;
    public int n1, n2, m1, m2;
    ImageView imageView;
    Point finger_p,right_p,left_p,point_p,finger_point;
    float right_eyeX, right_eyeY, left_eyeX,left_eyeY;
    float pointX,pointY,p_pointX,p_pointY;
    float averX,averY;
    List<Point> facearrays=new ArrayList<>();
    List<Point> pointarrays=new ArrayList<>();
    private org.opencv.core.Rect[] facesArray;


    private CaffeMobile caffeMobile;                                             //from CaffeMobile.java类
    File sdcard = Environment.getExternalStorageDirectory();                       //外部存储路径
    String modelDir = sdcard.getAbsolutePath() + "/caffe_mobile/bvlc_alexnet";
    String modelProto = modelDir + "/deployH.prototxt";                          //???
    String modelBinary = modelDir + "/bvlc_alexnet.caffemodel";          //caffe网络模型
    private byte[] image = new byte[90000];


    Bitmap  bitmap;
    Bitmap picture;

    private List<MatOfPoint> contours;// =   new ArrayList<MatOfPoint>();
    private Mat hierachy;//=new Mat();


    private static Bitmap cursorBitmap = null;
    private static final Matrix _matrix = new Matrix();
  //  private static final Paint _paint = new Paint(Paint.ANTI_ALIAS_FLAG);


    //装载jni库文件    即CaffeMobile.java 中函数的具体实现
    static {

        System.loadLibrary("caffe");
        System.loadLibrary("caffe_jni");
    }

    String img;
    /**
     * ATTENTION: This was auto-generated to implement the App Indexing API.
     * See https://g.co/AppIndexing/AndroidStudio for more information.
     */
    private GoogleApiClient client;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        //  setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);     //设置横屏
       // requestWindowFeature(Window.FEATURE_NO_TITLE);//无标题
        setContentView(R.layout.activity_main);   //设置显示界面
        cursorBitmap= BitmapFactory.decodeResource(getResources(), R.drawable.cursor);

        previewWidth = 240;           //预览像素宽高
        previewHeight = 320;
        sizeX = (float) 1.5;            //坐标缩放比例
        sizeY = (float) 2.2;


        caffeMobile = new CaffeMobile();                                     //创建caffeMobile实例对象
        caffeMobile.setNumThreads(4);                                        //创建4个线程进行计算
        caffeMobile.loadModel(modelProto, modelBinary);                       //导入模型

      //  float[] meanValues = {104, 117, 123};
        float[] meanValues = {88,91,96};  // 104,117,123     //88,91,96
        caffeMobile.setMean(meanValues);


        //  imageView=(ImageView)findViewById(R.id.ivCaptured);
/**
 *  SurfaceView1摄像头预览
 */
        mSurfaceview1 = (SurfaceView) findViewById(R.id.surfaceView1);
        // 设置该Surface不需要自己维护缓冲区
        mSurfaceview1.getHolder().setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        // mSurfaceview1.setAlpha((float)0.1);
        // 获得SurfaceView的SurfaceHolder
        mSurfaceHolder1 = mSurfaceview1.getHolder();
        // 为surfaceHolder添加一个回调监听器
        mSurfaceHolder1.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceChanged(SurfaceHolder holder, int format,
                                       int width, int height) {
            }

            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                // 打开摄像头
                initCamera();

            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                // 如果camera不为null ,释放摄像头
                if (mCamera != null) {
                    if (isPreview) mCamera.stopPreview();
                    mCamera.release();
                    mCamera = null;
                }
            }
        });


/**
 * SurfaceView2画布
 */
        mSurfaceview2 = (SurfaceView) findViewById(R.id.surfaceView2);
        mSurfaceview2.setZOrderOnTop(true);
        // 设置该Surface不需要自己维护缓冲区
        mSurfaceview2.getHolder().setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        // 获得SurfaceView的SurfaceHolder
        mSurfaceHolder2 = mSurfaceview2.getHolder();
        mSurfaceHolder2.setFormat(PixelFormat.TRANSLUCENT);
        // 为surfaceHolder添加一个回调监听器
        mSurfaceHolder2.addCallback(new SurfaceHolder.Callback() {
            @Override
            public void surfaceCreated(SurfaceHolder holder) {
                drawThread = new Thread(DrawThread);
                if (isPreview && !isRun) {
                    isRun = true;
                    drawThread.start();
                }
            }

            @Override
            public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {

            }

            @Override
            public void surfaceDestroyed(SurfaceHolder holder) {
                isRun = false;
            }

        });


        // ATTENTION: This was auto-generated to implement the App Indexing API.
        // See https://g.co/AppIndexing/AndroidStudio for more information.
        client = new GoogleApiClient.Builder(this).addApi(AppIndex.API).build();
    }

    Runnable DrawThread = new Runnable() //class DrawThread extends Thread
    {
        @Override
        public void run() {
            SurfaceHolder holder = mSurfaceHolder2;
            while (isRun & isPreview) {
                Canvas c = null;
                try {
                    synchronized (holder) {
                        c = holder.lockCanvas();//锁定画布，一般在锁定后就可以通过其返回的画布对象Canvas，在其上面画图等操作了。
                        c.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);//设置画布背景颜色

                        Paint p = new Paint(); //创建画笔p
                        p.setAntiAlias(true);
                        p.setColor(Color.GREEN);
                        p.setTextSize(30);
                        p.setStrokeWidth(5);
                        c.drawText("FPS:" + FPS, 300, 20, p);

                        Paint p1 = new Paint(); //创建画笔
                        p1.setAntiAlias(true);
                        p1.setColor(Color.BLUE);
                        p1.setTextSize(30);
                        p1.setStrokeWidth(5);

                        Paint p2 = new Paint(); //创建画笔
                        p2.setAntiAlias(true);
                        p2.setColor(Color.BLACK);
                        p2.setStrokeWidth(5);


                        Paint p3 = new Paint(); //创建画笔p
                        p3.setAntiAlias(true);
                        p3.setColor(Color.YELLOW);
                        p3.setStrokeWidth(5);
                        p3.setAlpha(220);

                        Paint p4 = new Paint(); //创建画笔p
                        p4.setAntiAlias(true);
                        p4.setColor(Color.YELLOW);
                        p4.setAlpha(180);

                        Paint p5 = new Paint(); //创建画笔p
                        p5.setAntiAlias(true);


                        p5.setColor(Color.YELLOW);
                        p5.setTextSize(30);
                      //  p5.setAlpha(140);
                        //  c.drawLine(0, 288, 984, 288, p2);
                        //  c.drawLine(0, 576, 984, 576, p2);
                        //    c.drawLine(0, 864, 984, 864, p2);
                        //  c.drawLine(0, 1152, 984, 1152, p2);

                        Paint _paint = new Paint(Paint.ANTI_ALIAS_FLAG);

                        c.drawLine(0, 360, 984, 360, p2);
                        c.drawLine(0, 720, 984, 720, p2);
                        c.drawLine(0, 1080, 984, 1080, p2);

                        c.drawLine(328, 0, 328, 1440, p2);
                        c.drawLine(656, 0, 656, 1440, p2);



                        c.drawLine(328, 0, 328, 360, p);
                        c.drawLine(0, 360, 328 , 360, p);
                        c.drawCircle(screenWidth, screenHeight, 20, p);      //screenWidth=1440   screenHeight=984
                        c.drawCircle(0, 0, 20, p);
                     //   c.drawBitmap(cursorBitmap,328,360,_paint);


                        if (pointsArray.width!=0||facesArray.length>0) {


                            newfaceArray = facesArray[0];
                            //  c.drawCircle( newfaceArray.x*X_SCALE, newfaceArray.y*Y_SCALE,20,p);

                            c.drawLine((newfaceArray.x) * X_SCALE, newfaceArray.y * Y_SCALE,
                                    ((newfaceArray.x + newfaceArray.width)) * X_SCALE, newfaceArray.y * Y_SCALE, p1);
                            c.drawLine(((newfaceArray.x + newfaceArray.width)) * X_SCALE, newfaceArray.y * Y_SCALE,
                                    ((newfaceArray.x + newfaceArray.width)) * X_SCALE, (newfaceArray.y + newfaceArray.height) * Y_SCALE, p1);
                            c.drawLine(((newfaceArray.x + newfaceArray.width)) * X_SCALE, (newfaceArray.y + newfaceArray.height) * Y_SCALE,
                                    (newfaceArray.x) * X_SCALE, (newfaceArray.y + newfaceArray.height) * Y_SCALE, p1);
                            c.drawLine((newfaceArray.x) * X_SCALE, (newfaceArray.y + newfaceArray.height) * Y_SCALE,
                                    (newfaceArray.x) * X_SCALE, newfaceArray.y * Y_SCALE, p1);


                            int width, height;
                            width = newfaceArray.width;
                            height = newfaceArray.height;
                            right_eyeX = Math.round(0.68 * width + newfaceArray.x);
                            right_eyeY = Math.round(0.38 * height + newfaceArray.y);
                            left_eyeX = Math.round(0.30 * width + newfaceArray.x);
                            left_eyeY = Math.round(0.38 * height + newfaceArray.y);
                            // c.drawCircle(right_eyeX * X_SCALE, right_eyeY * Y_SCALE, 20, p);

                            right_p = new Point(right_eyeX, right_eyeY);
                            facearrays.add(right_p);
                            if (facearrays.size() > 9) {

                                if (Math.sqrt((facearrays.get(9).x - facearrays.get(8).x) * (facearrays.get(9).x - facearrays.get(8).x) + (facearrays.get(9).y - facearrays.get(8).y) * (facearrays.get(9).y - facearrays.get(8).y)) > 30) {
                                    c.drawCircle(averX * X_SCALE, averY * Y_SCALE, 20, p1);
                                    facearrays.remove(9);
                                } else {

                                    float sumX = 0, sumY = 0;
                                    for (int i = 0; i < facearrays.size(); i++) {
                                        sumX = (float) (sumX + facearrays.get(i).x);
                                        sumY = (float) (sumY + facearrays.get(i).y);

                                    }
                                    averX = sumX / 10;
                                    averY = sumY / 10;
                                    c.drawCircle(averX * X_SCALE, averY * Y_SCALE, 20, p1);
                                    facearrays.remove(0);
                                }
                            }


                            newpointArray = pointsArray;
                            //  c.drawCircle( newfaceArray.x*X_SCALE, newfaceArray.y*Y_SCALE,20,p);

                            c.drawLine((newpointArray.x) * X_SCALE, newpointArray.y * Y_SCALE,
                                    ((newpointArray.x + newpointArray.width)) * X_SCALE, newpointArray.y * Y_SCALE, p);
                            c.drawLine(((newpointArray.x + newpointArray.width)) * X_SCALE, newpointArray.y * Y_SCALE,
                                    ((newpointArray.x + newpointArray.width)) * X_SCALE, (newpointArray.y + newpointArray.height) * Y_SCALE, p);
                            c.drawLine(((newpointArray.x + newpointArray.width)) * X_SCALE, (newpointArray.y + newpointArray.height) * Y_SCALE,
                                    (newpointArray.x) * X_SCALE, (newpointArray.y + newpointArray.height) * Y_SCALE, p);
                            c.drawLine((newpointArray.x) * X_SCALE, (newpointArray.y + newpointArray.height) * Y_SCALE,
                                    (newpointArray.x) * X_SCALE, newpointArray.y * Y_SCALE, p);

                              p_pointX=(float)(newpointArray.x+0.3*newpointArray.width);
                              p_pointY=newpointArray.y;
                            c.drawCircle(p_pointX * X_SCALE, p_pointY * Y_SCALE, 20, p);


/*
                            point_p=new Point(pointX,pointY);
                            pointarrays.add(point_p);
                            float p_averX=0,p_averY=0;
                            if(pointarrays.size()>9){
                                float p_sumX=0,p_sumY=0;
                                for(int i=0;i<pointarrays.size();i++){
                                    p_sumX=(float)(p_sumX+pointarrays.get(i).x);
                                    p_sumY=(float)(p_sumY+pointarrays.get(i).y);

                                }
                                p_averX=p_sumX/10;
                                p_averY=p_sumY/10;
                                  c.drawCircle(p_averX*X_SCALE,p_averY*Y_SCALE,20,p);
                                pointarrays.remove(0);

                            }
*/



                            // finger_p=mouse_finger(pointX,pointY,averX,averY);
                            finger_p = mouse_finger(p_pointX, p_pointY, right_eyeX, right_eyeY);
                            float a = (float) finger_p.x;
                            float b = (float) finger_p.y;
                            if (a <= 0) {
                                a = 0;
                            }
                            if (b <= 0) {
                                b = 0;
                            }
                            if (a >= 984) {
                                a = 983;
                            }
                            if (b >= 1440) {
                                b = 1439;
                            }

                            //c.drawCircle(a, b, 20, p1);


                            finger_point=new Point(a,b);
                            pointarrays.add(finger_point);
                        //    c.drawCircle((int)(pointarrays.get(0).x), (int)(pointarrays.get(0).y) , 15, p);
                            if (pointarrays.size() > 4) {
                                //opls_fitting(pointarrays);
                                linear_opls_fitting(pointarrays);
                                Distance = Point_distance(pointarrays);
                              //  c.drawCircle((int)(pointarrays.get(3).x) * X_SCALE, (int)(pointarrays.get(3).y) * Y_SCALE, 15, p);
                                if (Distance > 10000 || Distance < 5) {
                                  //  c.drawCircle((int)(pointarrays.get(3).x) , (int)(pointarrays.get(3).y) , 20, p5);
                                    c.drawBitmap(cursorBitmap,(int)(pointarrays.get(3).x),(int) (pointarrays.get(3).y),_paint);
                                    pointarrays.remove(4);

                                } else {
 /*
                                    c.drawCircle(m1 , (n1) , 20, p5);
                                    Thread.sleep(15);
                                    c.drawCircle(m2 , (n2) , 20, p5);
                                    Thread.sleep(15);
                                   c.drawCircle((int)(pointarrays.get(2).x),(int) (pointarrays.get(2).y) , 20, p5);
                                    Thread.sleep(15);


                                    c.drawCircle(w1, (z1) , 20, p4);
                                    Thread.sleep(15);
                                    c.drawCircle(w2, (z2) , 20, p4);
                                    Thread.sleep(15);
                                    c.drawCircle((int)(pointarrays.get(3).x), (int)(pointarrays.get(3).y) , 20, p4);
                                    Thread.sleep(15);


                                    c.drawCircle(x1 , (y1) , 20, p3);
                                    Thread.sleep(15);
                                    c.drawCircle(x2 , (y2) , 20, p3);
                                    Thread.sleep(15);

                                    c.drawCircle((int)(pointarrays.get(4).x) ,(int) (pointarrays.get(4).y) , 20, p3);
                                    */
                                    c.drawBitmap(cursorBitmap,(int)(pointarrays.get(4).x),(int) (pointarrays.get(4).y),_paint);
                                    pointarrays.remove(0);
                                }


                                //c.drawCircle(a, b, 20, p1);


                            }
                          //  c.drawBitmap(cursorBitmap,328,360,_paint);
                            if((pointarrays.get(3).x)<328&&(pointarrays.get(3).y)<360){
                                num++;
                            }
                            if(num>8){
                                startActivity(new Intent(MainActivity.this,Main2Activity.class));
                            }


                          //  _matrix.reset();
                           // _matrix.postTranslate((int)pointarrays.get(3).x,(int)pointarrays.get(3).y);
                           // _matrix.postTranslate(300,300);
                          //  c.drawBitmap(bitmap, _matrix, _paint);

                        }

                        Thread.sleep(10);//睡眠时间为0.01秒
                        //c.restore();
                    }
                } catch (Exception e) {
                    // TODO: handle exception
                    e.printStackTrace();
                } finally {
                    if (c != null) {
                        holder.unlockCanvasAndPost(c);//结束锁定画图，并提交改变。
                    }
                }
            }
        }
    };


    /**
     * 启动摄像头
     */
    private void initCamera() {
        if (!isPreview) {
            // 此处默认打开后置摄像头。
            // 通过传入参数可以打开前置摄像头
            mCamera = Camera.open(1);  //①
            mCamera.setDisplayOrientation(90);
            //mCamera.setDisplayOrientation(0);
        }
        if (mCamera != null && !isPreview) {
            try {

                Camera.Parameters parameters = mCamera.getParameters();
                // parameters.set("auto-whitebalance-lock","true");
                //  pa[rameters.set("auto-exposure-lock","true");
                //  parameters.set("max-exposure-compensation",5);
                //  parameters.set("min-exposure-compensation",-5);
                //  parameters.set("exposure-compensation",4);

                parameters.setExposureCompensation(3);
                parameters.setAutoExposureLock(true);
                // parameters.setAutoWhiteBalanceLock(true);
                parameters.getAutoExposureLock();
                parameters.getExposureCompensation();
                parameters.getExposureCompensationStep();
                parameters.getMaxExposureCompensation();
                parameters.getMinExposureCompensation();
                parameters.isAutoExposureLockSupported();
                parameters.getFocusMode();

                //  parameters.setFocusMode("macro");
                //  parameters.getFocusMode();
                // float[] output=new float[3];
                // parameters.getFocusDistances(output);

                screenWidth = mSurfaceview1.getWidth();  //屏幕宽度
                screenHeight = mSurfaceview1.getHeight();   //屏幕高度
                X_SCALE = screenWidth / (float) previewWidth;
                Y_SCALE = screenHeight / (float) previewHeight;
                parameters.setPreviewSize(previewHeight, previewWidth);

                // List<Integer> formatsList = parameters.getSupportedPreviewFormats();
                mCamera.setPreviewDisplay(mSurfaceHolder1);  //②
                mCamera.setPreviewCallback(this);
                mCamera.setParameters(parameters);
                parameters = mCamera.getParameters();
                mCamera.startPreview();                      // 开始预览

            } catch (Exception e) {
                e.printStackTrace();
            }
            isPreview = true;
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        mSurfaceview1.setVisibility(View.VISIBLE);
        //  OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, mLoaderCallback);
        // imageView.setImageBitmap(bitmap);
        mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        initializeOpenCVDependencies();

    }

    //OpenCV类库加载并初始化成功后的回调函数，在此我们不进行任何操作
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i("OpenCV", "OpenCV loaded successfully");
                    ////这里可以进行一些操作
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    @Override
    public void onPause() {
        super.onPause();
        if (isPreview) {
            mCamera.setPreviewCallback(null);
            mCamera.stopPreview();
            mCamera.release();
            mCamera = null;
            isPreview = false;
        }

        mSurfaceview1.setVisibility(View.INVISIBLE);

    }

    /**
     * 载入分类器
     */
    private void initializeOpenCVDependencies() {

        try {
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);
            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();
            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());

        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }
    }

    /**
     * onPreviewFrame
     *
     * @param bytes
     * @param camera
     */
    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        // long startTime = SystemClock.uptimeMillis();
        Date curTime = new Date(System.currentTimeMillis());//获取当前时间
        double fps = 1000 / ((long) curTime.getTime() - (long) lastTime.getTime());
        FPS = String.valueOf(fps);
        lastTime = curTime;

        yuvImg = new Mat(previewWidth * 3 / 2, previewHeight, CvType.CV_8UC1);

        yuvImg.put(0, 0, bytes);
        rotateImg=new Mat();
        Imgproc.cvtColor(yuvImg, rotateImg, Imgproc.COLOR_YUV2RGB_NV21);//COLOR_YUV2GRAY_NV21);//COLOR_YUV2RGB_I420);//
        Mat testImage = new Mat();
        rotateImg.copyTo(testImage);



        Mat image=new Mat();
        transpose(rotateImg,image);
        flip(image, image, 1);
        flip(image, image, 0);
        //flip(rotateImg, rotateImg, 1);
        //  byte []byte2=new byte[bytes.length];
        //  yuvImg.get(0,0,byte2);

        //  convertYUV(yuvImg,240);

        Mat ROIImage=new Mat();
        Rect R=new Rect(0,80,240,240);
        image.submat(R).copyTo(ROIImage);
        Size s=new Size(240,240);
        Imgproc.resize(ROIImage,ROIImage,s);

        yuvImg2=new Mat();
        Imgproc.cvtColor(ROIImage,yuvImg2,  Imgproc.COLOR_RGB2YUV_YV12);//COLOR_YUV2RGB_I420);//
        byte []byte2=new byte[360*240];
        yuvImg2.get(0, 0, byte2);
        byte []byte3=new byte[byte2.length];
        swapYV12toNV21(byte2,byte3,240,240);


        //  saveYuv(byte3,camera,240,240);


        //  long startTime = SystemClock.uptimeMillis();
        //  float [][]data=caffeMobile.extractFeatures(img,"score");

        //   float [][]data=caffeMobile.extractFeatures(aa,"score");
        // Log.d(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));

        float[][] data = caffeMobile.extractFeatures(byte3, "score");
        //     Log.e(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));
        // bitmap=BinaryArgmax(data);
//   imageView.setImageBitmap(bitmap);


        //    Bitmap bit=BinaryArgmax(data,320,320);
        //  saveBitmap(bit);

        Mat newResult = MatArgmax(data,240,240);

        contours = new ArrayList<>();
        hierachy = new Mat();
        Mat ContoursMat=new Mat();
        newResult.copyTo(ContoursMat);

        Imgproc.findContours(ContoursMat, contours, hierachy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);



        double maxArea = -1;
        int maxAreaIdx = -1;
        //org.opencv.core. Rect[] rect=new Rect[contours.size()];
        for (int i = 0; i < contours.size(); i++) {
            double contourarea = Imgproc.contourArea(contours.get(i));
            if (contourarea > maxArea) {
                maxArea = contourarea;
                maxAreaIdx = i;

            }


        }



        if (maxAreaIdx != -1) {
             if(Imgproc.contourArea(contours.get(maxAreaIdx))>1000&&Imgproc.contourArea(contours.get(maxAreaIdx))<2500) {
            pointsArray = Imgproc.boundingRect(contours.get(maxAreaIdx));
            int temp=pointsArray.x;
            for(int i=pointsArray.x;i<pointsArray.x+pointsArray.width;i++){
                if(newResult.get(pointsArray.y,i)[0]==255){
                    temp=i;
                    break;
                }
            }

            pointX=temp;
            pointY=pointsArray.y;


            pointX=(int)(1*pointX+R.x);
            pointY=(int)(1*pointY+R.y);




            pointsArray.x=(int)(1*pointsArray.x+R.x);
            pointsArray.y=(int)(1*pointsArray.y+R.y);
            pointsArray.width=(int)(1*pointsArray.width);
            pointsArray.height=(int)(1*pointsArray.height);


               }



             else {
                 if (pointsArray != null) {
                     pointsArray.width = 0;
                     pointsArray.height = 0;
                 }
             }
        }

        else{
            if(pointsArray!=null) {
                pointsArray.width = 0;
                pointsArray.height = 0;
            }

        }



        Core.copyMakeBorder(testImage,testImage,(previewHeight-previewWidth)/2,(previewHeight-previewWidth)/2,0,0,Core.BORDER_CONSTANT);//放大画布1920*1920


        Point center = new Point(previewHeight/2,previewHeight/2 );  // 旋转中心
        double angle = -90;  // 旋转角度
        double scale =1; // 缩放尺度
        rotateMat = Imgproc.getRotationMatrix2D(center, angle, scale);//计算旋转矩阵
        flip(testImage,testImage,1);
        Imgproc.warpAffine(testImage,testImage,rotateMat, testImage.size());//旋转

        org.opencv.core.Rect Roi=new org.opencv.core.Rect((previewHeight-previewWidth)/2,0,previewWidth,previewHeight);//去黑边
        Mat detectImage=new Mat();
        testImage.submat(Roi).copyTo(detectImage);

        face_recognition(detectImage);

        // Log.e(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));

    }


    /**
     *  手势检测
     * @param aInputFrame
     */


    public void face_recognition(Mat aInputFrame)
    {
        //org.opencv.core.Rect fa=new org.opencv.core.Rect(10,10,50,50);
        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale( aInputFrame, faces, 1.1, 2, 2,
                    new Size(absolutePointSize, absolutePointSize), new Size());
        }
        // If there are any faces found, draw a rectangle around it
        if(faces.height()!=0) {
            facesArray = faces.toArray();
        }

    }



    /**
     * 最小二乘法拟合轨迹 + 拟合曲线差值
     *
     * @param CursorPos
     */
    public void opls_fitting(List<Point> CursorPos) {
        //(1) 求多项式参数
        double a0, a1, b0, u0, u1, u2;
        a0 = 0;
        u0 = 0;
        for (int i = 0; i < CursorPos.size(); i++) {
            a0 += CursorPos.get(i).x;
            u0 += CursorPos.get(i).y;
        }
        a0 /= 5.0;
        u0 /= 5.0;
        double t1 = 0.0, t2 = 0.0;
        u1 = 0.0;
        for (int i = 0; i < CursorPos.size(); i++) {
            t1 += CursorPos.get(i).x * pow(CursorPos.get(i).x - a0, 2);
            t2 += pow(CursorPos.get(i).x - a0, 2);
            u1 += CursorPos.get(i).y * (CursorPos.get(i).x - a0);
        }
        if (t2 == 0) return;
        a1 = t1 / t2;
        b0 = t2 / 5.0;
        u1 = u1 / t2;
        u2 = 0.0;
        t1 = 0.0;
        t2 = 0.0;
        for (int i = 0; i < CursorPos.size(); i++) {
            t1 += CursorPos.get(i).y * ((CursorPos.get(i).x - a1) * (CursorPos.get(i).x - a0) - b0);
            t2 += pow((CursorPos.get(i).x - a1) * (CursorPos.get(i).x - a0) - b0, 2);
        }
        if (t2 == 0) return;
        u2 = t1 / t2;
        //(2) 拟合校正CursorPos[4]
        int y;
        int x = (int)CursorPos.get(4).x;
        y = (int) Math.round(u0 + u1 * (x - a0) + u2 * ((x - a1) * (x - a0) - b0));


        float height = mSurfaceview1.getHeight();
        int hsys = (int) (height / (Y_SCALE));
        if (y > hsys - 1) y = hsys - 1;//防止越界
        CursorPos.get(4).y = y;


        x1 = (int)(pointarrays.get(3).x + (pointarrays.get(4).x - pointarrays.get(3).x) / 3);
        x2 = (int)(pointarrays.get(3).x + (pointarrays.get(4).x - pointarrays.get(3).x) / 3 * 2);
        w1 = (int)(pointarrays.get(2).x + (pointarrays.get(3).x - pointarrays.get(2).x) / 3);
        w2 = (int)(pointarrays.get(2).x + (pointarrays.get(3).x - pointarrays.get(2).x) / 3 * 2);
        m1 = (int)(pointarrays.get(1).x + (pointarrays.get(2).x - pointarrays.get(1).x) / 3);
        m2 = (int)(pointarrays.get(1).x + (pointarrays.get(2).x - pointarrays.get(1).x) / 3 * 2);

        y1 = (int) Math.round(u0 + u1 * (x1 - a0) + u2 * ((x1 - a1) * (x1 - a0) - b0));
        y2 = (int) Math.round(u0 + u1 * (x2 - a0) + u2 * ((x2 - a1) * (x2 - a0) - b0));
        z1 = (int) Math.round(u0 + u1 * (w1 - a0) + u2 * ((w1 - a1) * (w1 - a0) - b0));
        z2 = (int) Math.round(u0 + u1 * (w2 - a0) + u2 * ((w2 - a1) * (w2 - a0) - b0));
        n1 = (int) Math.round(u0 + u1 * (m1 - a0) + u2 * ((m1 - a1) * (m1 - a0) - b0));
        n2 = (int) Math.round(u0 + u1 * (m2 - a0) + u2 * ((m2 - a1) * (m2 - a0) - b0));


    }



    public void linear_opls_fitting(List<Point> CursorPos) {
        x1 = (int)(pointarrays.get(3).x + (pointarrays.get(4).x - pointarrays.get(3).x) / 3);
        x2 = (int)(pointarrays.get(3).x + (pointarrays.get(4).x - pointarrays.get(3).x) / 3 * 2);
        w1 = (int)(pointarrays.get(2).x + (pointarrays.get(3).x - pointarrays.get(2).x) / 3);
        w2 = (int)(pointarrays.get(2).x + (pointarrays.get(3).x - pointarrays.get(2).x) / 3 * 2);
        m1 = (int)(pointarrays.get(1).x + (pointarrays.get(2).x - pointarrays.get(1).x) / 3);
        m2 = (int)(pointarrays.get(1).x + (pointarrays.get(2).x - pointarrays.get(1).x) / 3 * 2);

        y1 =  (int)(pointarrays.get(3).y + (pointarrays.get(4).y - pointarrays.get(3).y) / 3);
        y2 =  (int)(pointarrays.get(3).y + (pointarrays.get(4).y - pointarrays.get(3).y) / 3 * 2);
        z1 =  (int)(pointarrays.get(2).y + (pointarrays.get(3).y - pointarrays.get(2).y) / 3);
        z2 = (int)(pointarrays.get(2).y + (pointarrays.get(3).y - pointarrays.get(2).y) / 3 * 2);
        n1 = (int)(pointarrays.get(1).y + (pointarrays.get(2).y - pointarrays.get(1).y) / 3);
        n2 =(int)(pointarrays.get(1).y + (pointarrays.get(2).y - pointarrays.get(1).y) / 3 * 2);
    }

    /**
     * 距离函数
     *
     * @param array
     * @return 两检测点间距离
     */
    public double Rect_distance(List<Rect> array) {
        return Math.sqrt((array.get(4).x - array.get(3).x) * (array.get(4).x - array.get(3).x) + (array.get(4).y - array.get(3).y) * (array.get(4).y - array.get(3).y));
    }


    public double Point_distance(List<Point> array) {
        return Math.sqrt((array.get(4).x - array.get(3).x) * (array.get(4).x - array.get(3).x) + (array.get(4).y - array.get(3).y) * (array.get(4).y - array.get(3).y));
    }
    private Bitmap BinaryArgmax(float[][] data,int width,int height) {
        int Length = data[0].length;
        int length = Length / 2;
        int[] pixels = new int[length];
        Bitmap newBmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        for (int i = 0; i < length; i++) {
            if (data[0][i] < data[0][i + length]) {
                pixels[i] = Color.rgb(255, 255, 255);      //黑色
            } else {
                pixels[i] = Color.rgb(0, 0, 0);    //白色
            }

        }
        newBmp.setPixels(pixels, 0,width, 0, 0, width, height);
        return newBmp;

    }

    public void saveBitmap(Bitmap mBitmap) {
        File f = new File("/sdcard/" + System.currentTimeMillis() + ".png");
        try {
            f.createNewFile();
        } catch (IOException e) {
            //Toast("在保存图片时出错：" + e.toString());
        }
        FileOutputStream fOut = null;
        try {
            fOut = new FileOutputStream(f);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        mBitmap.compress(Bitmap.CompressFormat.PNG, 100, fOut);
        try {
            fOut.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        try {
            fOut.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    private Mat MatArgmax(float[][] data,int width,int height) {
        int Length = data[0].length;
        int length = Length / 2;
        double[] pixels = new double[length];
        Mat newMat = new Mat(width, height, CvType.CV_8UC1);
        for (int i = 0; i < length; i++) {
            if (data[0][i] < data[0][i + length]) {
                pixels[i] = 255.0;      //白色
            } else {
                pixels[i] = 0.0;    //黑色
            }

        }
        newMat.put(0, 0, pixels);

/*
        int num=0;
        long startTime;
        startTime = SystemClock.uptimeMillis();
        for(int i=0;i<pixels.length;i++){
            if(pixels[i]==0.0){
                num=i;
                break;
            }
        }
        Log.d(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));
*/

        return newMat;

    }

    private void convertYUV(Mat scr, int Y_height) {
        double[] temp1;
        double[] temp2;
        for (int j = Y_height; j < scr.height(); j++) {
            for (int i = 0; i < scr.width(); i = i + 2) {
                temp1 = scr.get(j, i);
                temp2 = scr.get(j, i + 1);
                scr.put(j, i + 1, temp1);
                scr.put(j, i, temp2);
            }
        }

    }



    void swapYV12toNV21(byte[] yv12bytes, byte[] nv12bytes, int width,int height)
    {
        int nLenY = width * height;
        int nLenU = nLenY / 4;
        System.arraycopy(yv12bytes, 0, nv12bytes, 0, width * height);
        for (int i = 0; i < nLenU; i++) {
            //  nv12bytes[nLenY + 2 * i + 1] =yv12bytes[nLenY + nLenU + i];
            //  nv12bytes[nLenY + 2 * i] = yv12bytes[nLenY + i];
            nv12bytes[nLenY + 2 * i + 1] =yv12bytes[nLenY + i];
            nv12bytes[nLenY + 2 * i] = yv12bytes[nLenY + nLenU + i];
        }
    }

    private void NV21ToNV12(byte[] nv21, byte[] nv12, int width, int height) {
        if (nv21 == null || nv12 == null) return;
        int framesize = width * height;
        int i = 0, j = 0;
        System.arraycopy(nv21, 0, nv12, 0, framesize);
        for (i = 0; i < framesize; i++) {
            nv12[i] = nv21[i];
        }
        for (j = 0; j < framesize / 2; j += 2) {
            nv12[framesize + j - 1] = nv21[j + framesize];
        }
        for (j = 0; j < framesize / 2; j += 2) {
            nv12[framesize + j] = nv21[j + framesize - 1];
        }
    }




    private void saveYuv(byte[] bytes, Camera camera,int width,int height) {
        File sdcard = Environment.getExternalStorageDirectory();
        String modelDir = sdcard.getAbsolutePath() + "/Data_Age";
        FileOutputStream outStream = null;
        try {
            YuvImage yuvimage = new YuvImage(bytes, camera.getParameters().getPreviewFormat(),width, height, null);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            yuvimage.compressToJpeg(new android.graphics.Rect(0, 0, width, height), 100, baos);

            outStream = new FileOutputStream(modelDir + System.currentTimeMillis() + ".jpg");
            outStream.write(baos.toByteArray());
            outStream.flush();
            outStream.close();


        } catch (FileNotFoundException e) {
            e.printStackTrace();
            Log.i("yy", "1111111111111111111");
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
        }
    }


    private Point mouse_finger(float pointX,float pointY,float eyeX,float eyeY) {
        double fore_l1 = 50;//手距离屏幕的距离
        double fore_l2 = 70;//人脸距离屏幕的距离
        double l2 =50 ;
        double l1 = 25;//【改】
        double l3 = 41;//在l1距离水平的视角         //需修改     28  52  43
        double l4 = 37;//在l1距离垂直的视角          //需修改
        double a = 182;//采集画面的宽度的一半              weight=984    120   208
        double b =160;//采集画面的长度的一半              height=1476
        double pix = 149;//当前电脑每厘米对应的像素值      //需修改
        double w_face = 17;//人脸宽，单位厘米(用haar测得的人脸宽会比实际大一些,15-》17)

        double dis_x = (a - eyeX)*l2 -(a - pointX)*l1;
        double dis_y =(b - eyeY)*l2 - (b - pointY)*l1;

//2017.3.22  jun 根据图像中人脸宽设定人脸与屏幕的距离l2
        //修改1

/*
        int facewidth = newfaceArray.width;
        //修改2
        l2 = (416*w_face*l1) / (facewidth * l3);
        l2 = l2 > 50 ? 50 : l2;//防止变化过大,此处设定人脸与屏幕的距离在[60,130]厘米内
        l2 = l2 < 30 ? 30 : l2;
*/
        double coe_x1 = (pix*l2*((l3 / 2) / l1) / (a*(l2 - l1)));//加了人眼后的系数【zm】
        double coe_y1 = pix*(l2 *((l4 / 2) / l1)) / (b*(l2 - l1));//加了人眼后的系数【zm】
        //修改3
        int width =1491;    //416
        int x3= width / 2 - (int)(l2*((l3 / 2) / l1) / a*(a - eyeX)*pix);//眼睛距离屏幕最左端水平距离【zm0526】//像素
        double cam_dis = 1.5*pix;//摄像头距离屏幕最上端的距离

        Point p=new Point();
        p.x = Math.round(coe_x1*dis_x + x3);
        p.y =Math.round(Math.abs(Math.abs(coe_y1*dis_y - cam_dis) - (b - eyeY) / b*l2*((l4 / 2) / l1)*pix));

        return p;
    }

}