package org.firstinspires.ftc.teamcode;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.openftc.easyopencv.OpenCvPipeline;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class ArtifactCirclePipeline extends OpenCvPipeline {
    /*
     * Our working image buffers
     */
    Mat yCrCbMat = new Mat();
    Mat crMat = new Mat();
    Mat cbMat = new Mat();
    Mat thresholdMat = new Mat();
    Mat morphedThreshold = new Mat();
    Mat contoursOnPlainImageMat = new Mat();

    // purple artifacts
    Mat pCrMat = new Mat();
    Mat pCbMat = new Mat();
    Mat pThresholdMat = new Mat();
    Mat pMorphedThreshold = new Mat();

    // green artifacts
    Mat gCrMat = new Mat();
    Mat gCbMat = new Mat();
    Mat gThresholdMat = new Mat();
    Mat gMorphedThreshold = new Mat();


    /*
     * The elements we use for noise reduction
     */
    Mat erodeElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
    Mat dilateElement = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(6, 6));

    /*
     * Colors
     */
    static final Scalar TEAL = new Scalar(3, 148, 252);
    static final Scalar PURPLE = new Scalar(158, 52, 235);
    static final Scalar RED = new Scalar(255, 0, 0);
    static final Scalar GREEN = new Scalar(0, 255, 0);
    static final Scalar BLUE = new Scalar(0, 0, 255);

    static final int CONTOUR_LINE_THICKNESS = 2;

    /*
     * Some stuff to handle returning our various buffers
     */
    enum Stage
    {
        FINAL,
        Cb,
        Cr,
        pThresh,
        pMorph,
        gThresh,
        gMorph,
        CONTOURS;
    }

    Stage[] stages = Stage.values();

    // Keep track of what stage the viewport is showing
    int stageNum = 0;

    @Override
    public void onViewportTapped()
    {
        /*
         * Note that this method is invoked from the UI thread
         * so whatever we do here, we must do quickly.
         */

        int nextStageNum = stageNum + 1;

        if(nextStageNum >= stages.length)
        {
            nextStageNum = 0;
        }

        stageNum = nextStageNum;
    }

    @Override
    public Mat processFrame(Mat input)
    {
        /*
         * Run the image processing
         */
        analyzeArtifacts(input, findContours(input));

        switch (stages[stageNum])
        {
            case FINAL:
            {
                return input;
            }

            case Cb:
            {
                return cbMat;
            }

            case Cr:
            {
                return crMat;
            }

            case pThresh:
                return pThresholdMat;
            case pMorph:
                return pMorphedThreshold;
            case gThresh:
                return gThresholdMat;
            case gMorph:
                return gMorphedThreshold;

            case CONTOURS:
            {
                return contoursOnPlainImageMat;
            }
        }

        return input;
    }

    Artifacts findContours(Mat input)
    {
        /*
            public static final ColorRange ARTIFACT_GREEN = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 32,  50, 118),
            new Scalar(255, 105, 145)
    );

    public static final ColorRange ARTIFACT_PURPLE = new ColorRange(
            ColorSpace.YCrCb,
            new Scalar( 32, 135, 135),
            new Scalar(255, 155, 169)
    );
    */

        // A list we'll be using to store the contours we find
        Artifacts artifacts = new Artifacts();

        // Convert the input image to YCrCb color space, then extract the Cb and Cr channels
        Imgproc.cvtColor(input, yCrCbMat, Imgproc.COLOR_RGB2YCrCb);
        Core.extractChannel(yCrCbMat, crMat, 1);
        Core.extractChannel(yCrCbMat, cbMat, 2);

        // ------ get purple artifacts --------



        // get thresholds for cB and cR channels
        Core.inRange(crMat, new Scalar(135),new Scalar(200),pCrMat);
        Core.inRange(cbMat,new Scalar(135), new Scalar(175), pCbMat);

        // combine thresholds - if the pixel is in Cr AND Cb thresholds
        Core.bitwise_and(pCrMat,pCbMat,pThresholdMat);

        // erode and dilate
        morphMask(pThresholdMat, pMorphedThreshold);

        // Ok, now actually look for the contours! We only look for external contours.
        Imgproc.findContours(pMorphedThreshold, artifacts.purple, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        artifacts.purple.removeIf(contour -> Imgproc.contourArea(contour) <= 675);


        // ------- green artifacts ---------


        // Convert the input image to YCrCb color space, then extract the Cb and Cr channels

        // get thresholds for cB and cR channels

        //            new Scalar( 32,  50, 118),
        //            new Scalar(255, 105, 145)
        Core.inRange(crMat, new Scalar(50),new Scalar(105),gCrMat);
        Core.inRange(cbMat,new Scalar(118), new Scalar(145), gCbMat);

        // combine thresholds - if the pixel is in Cr AND Cb thresholds
        Core.bitwise_and(gCrMat,gCbMat,gThresholdMat);

        // erode and dilate
        morphMask(gThresholdMat, gMorphedThreshold);
        // Ok, now actually look for the contours! We only look for external contours.
        Imgproc.findContours(gMorphedThreshold, artifacts.green, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        // We do draw the contours we find, but not to the main input buffer.
        artifacts.green.removeIf(contour -> Imgproc.contourArea(contour) <= 675);

        input.copyTo(contoursOnPlainImageMat);
        Imgproc.drawContours(contoursOnPlainImageMat, artifacts.purple, -1, PURPLE, CONTOUR_LINE_THICKNESS, 8);
        Imgproc.drawContours(contoursOnPlainImageMat, artifacts.green, -1, GREEN, CONTOUR_LINE_THICKNESS, 8);

        return artifacts;
    }

    void morphMask(Mat input, Mat output)
    {
        /*
         * Apply some erosion and dilation for noise reduction
         */


        Imgproc.erode(input, output, erodeElement);
        Imgproc.erode(output, output, erodeElement);
        Imgproc.dilate(output, output, dilateElement);
        Imgproc.dilate(output, output, dilateElement);
    }

    void analyzeArtifacts(Mat input, Artifacts artifacts)
    {
        // Transform the contour to a different format
//        Point[] points = contour.toArray();
//        MatOfPoint2f contour2f = new MatOfPoint2f(contour.toArray());
        int i = 0;
        for (MatOfPoint contour : artifacts.purple) {
            MatOfPoint2f contour2f = new MatOfPoint2f();
            contour.convertTo(contour2f, CvType.CV_32F);

            // Fit minimum enclosing circle
            Point center = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(contour2f, center, radius);

            // Draw filled circle on mask
            Imgproc.circle(input, center,2, new Scalar(255,255,255), 2);
            Imgproc.putText(input,String.format("PURPLE: %.2f,%.2f",center.x,center.y),center,Imgproc.FONT_HERSHEY_PLAIN,1,new Scalar(255,0,2550));
            Imgproc.circle(input, center, (int) radius[0], new Scalar(255,0,255), 3);  // -1 = filled
            HashMap<Integer,Point> m = new HashMap<>();
            m.put(i,center);

            Artifacts.purplePoints.add(m);
            i++;
        }

        i = 0;
        for (MatOfPoint contour : artifacts.green) {
            MatOfPoint2f contour2f = new MatOfPoint2f();
            contour.convertTo(contour2f, CvType.CV_32F);

            // Fit minimum enclosing circle
            Point center = new Point();
            float[] radius = new float[1];
            Imgproc.minEnclosingCircle(contour2f, center, radius);

            // Draw filled circle on mask
            Imgproc.circle(input, center,2, new Scalar(255,255,255), 2);
            Imgproc.putText(input,String.format("GREEN: %.2f,%.2f",center.x,center.y),center,Imgproc.FONT_HERSHEY_PLAIN,1,new Scalar(0,255,0));
            Imgproc.circle(input, center, (int) radius[0], new Scalar(0,255,0), 3);  // -1 = filled

            HashMap<Integer,Point> m = new HashMap<>();
            m.put(i,center);
            Artifacts.greenPoints.add(m);
            i++;
        }


        // Do a rect fit to the contour, and draw it on the screen
//        RotatedRect rotatedRectFitToContour = Imgproc.minAreaRect(contour2f);
//        drawRotatedRect(rotatedRectFitToContour, input);
    }

    public static class Artifacts {
        ArrayList<MatOfPoint> purple;
        ArrayList<MatOfPoint> green;
        static ArrayList<HashMap<Integer,Point>> purplePoints;
        static ArrayList<HashMap<Integer,Point>> greenPoints;


        Artifacts() {
            purple = new ArrayList<>();
            green = new ArrayList<>();

            purplePoints = new ArrayList<>();
            greenPoints = new ArrayList<>();
        }

        public static ArrayList<HashMap<Integer,Point>> getArtifactsPurple() {
            return purplePoints;
        }

        public static ArrayList<HashMap<Integer,Point>> getArtifactsGreen() {
            return greenPoints;
        }
    }
}
