package opencv;

import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.highgui.Highgui;

import java.io.File;
import java.util.LinkedList;
import java.util.List;

public class StarMapSURFDetector {

	public static void main(String[] args) {

		File lib = null;
		String bitness = System.getProperty("sun.arch.data.model");

		if (bitness.endsWith("64")) {
			lib = new File("libs//x64//"
					+ System.mapLibraryName("opencv_java2411"));
		} else {
			lib = new File("libs//x86//"
					+ System.mapLibraryName("opencv_java2411"));
		}

		System.out.println(lib.getAbsolutePath());
		System.load(lib.getAbsolutePath());

		String smallPhoto = "images//ksmall3rot.png";
		String starMapBigPicture = "images//orj.png";

		System.out.println("Started....");
		System.out.println("Loading images...");
		Mat objectImage = Highgui.imread(smallPhoto,
				Highgui.CV_LOAD_IMAGE_GRAYSCALE);
		Mat sceneImage = Highgui.imread(starMapBigPicture,
				Highgui.CV_LOAD_IMAGE_GRAYSCALE);

		MatOfKeyPoint objectKeyPoints = new MatOfKeyPoint();
		FeatureDetector featureDetector = FeatureDetector
				.create(FeatureDetector.SURF);
		System.out.println("Detecting points...");
		featureDetector.detect(objectImage, objectKeyPoints);
		KeyPoint[] keypoints = objectKeyPoints.toArray();
		System.out.println(keypoints);

		MatOfKeyPoint objectDescriptors = new MatOfKeyPoint();
		DescriptorExtractor descriptorExtractor = DescriptorExtractor
				.create(DescriptorExtractor.SURF);
		System.out.println("Computing...");
		descriptorExtractor.compute(objectImage, objectKeyPoints,
				objectDescriptors);

		Mat outputImage = new Mat(objectImage.rows(), objectImage.cols(),
				Highgui.CV_LOAD_IMAGE_COLOR);
		Scalar newKeypointColor = new Scalar(255, 0, 0);

		Features2d.drawKeypoints(objectImage, objectKeyPoints, outputImage,
				newKeypointColor, 0);

		MatOfKeyPoint sceneKeyPoints = new MatOfKeyPoint();
		MatOfKeyPoint sceneDescriptors = new MatOfKeyPoint();
		featureDetector.detect(sceneImage, sceneKeyPoints);
		
		descriptorExtractor.compute(sceneImage, sceneKeyPoints,
				sceneDescriptors);

		Mat matchoutput = new Mat(sceneImage.rows() * 2, sceneImage.cols() * 2,
				Highgui.CV_LOAD_IMAGE_COLOR);
		Scalar matchestColor = new Scalar(0, 255, 0);

		List<MatOfDMatch> matches = new LinkedList<MatOfDMatch>();
		DescriptorMatcher descriptorMatcher = DescriptorMatcher
				.create(DescriptorMatcher.FLANNBASED);

		//knn match with descriptor
		descriptorMatcher.knnMatch(objectDescriptors, sceneDescriptors,
				matches, 4);

		System.out.println("Calculating...");
		LinkedList<DMatch> goodMatchesList = new LinkedList<DMatch>();

		
		float ratio = 0.8f;

		for (int i = 0; i < matches.size(); i++) {
			MatOfDMatch matofDMatch = matches.get(i);
			DMatch[] dmatcharray = matofDMatch.toArray();
			DMatch m1 = dmatcharray[0];
			DMatch m2 = dmatcharray[1];

			if (m1.distance <= m2.distance * ratio) {
				goodMatchesList.addLast(m1);

			}
		}

		if (goodMatchesList.size() >= 5) {
			System.out.println("Scene Found!!!");

			List<KeyPoint> objKeypointlist = objectKeyPoints.toList();
			List<KeyPoint> scnKeypointlist = sceneKeyPoints.toList();

			LinkedList<Point> objectPoints = new LinkedList<>();
			LinkedList<Point> scenePoints = new LinkedList<>();

			for (int i = 0; i < goodMatchesList.size(); i++) {
				objectPoints
						.addLast(objKeypointlist.get(goodMatchesList.get(i).queryIdx).pt);
				scenePoints
						.addLast(scnKeypointlist.get(goodMatchesList.get(i).trainIdx).pt);
			}

			MatOfPoint2f objMatOfPoint2f = new MatOfPoint2f();
			objMatOfPoint2f.fromList(objectPoints);
			MatOfPoint2f scnMatOfPoint2f = new MatOfPoint2f();
			scnMatOfPoint2f.fromList(scenePoints);

			Mat homography = Calib3d.findHomography(objMatOfPoint2f,
					scnMatOfPoint2f, Calib3d.RANSAC, 3);

			Mat corners = new Mat(4, 1, CvType.CV_32FC2);
			Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

			corners.put(0, 0, new double[] { 0, 0 });
			corners.put(1, 0, new double[] { objectImage.cols(), 0 });
			corners.put(2, 0, new double[] { objectImage.cols(),
					objectImage.rows() });
			corners.put(3, 0, new double[] { 0, objectImage.rows() });

			Core.perspectiveTransform(corners, scene_corners, homography);

			Mat img = Highgui.imread(starMapBigPicture, Highgui.CV_LOAD_IMAGE_COLOR);

			Core.line(img, new Point(scene_corners.get(0, 0)), new Point(
					scene_corners.get(1, 0)), new Scalar(0, 255, 0), 4);
			Core.line(img, new Point(scene_corners.get(1, 0)), new Point(
					scene_corners.get(2, 0)), new Scalar(0, 255, 0), 4);
			Core.line(img, new Point(scene_corners.get(2, 0)), new Point(
					scene_corners.get(3, 0)), new Scalar(0, 255, 0), 4);
			Core.line(img, new Point(scene_corners.get(3, 0)), new Point(
					scene_corners.get(0, 0)), new Scalar(0, 255, 0), 4);

			System.out.println("Drawing matches image...");
			MatOfDMatch goodMatches = new MatOfDMatch();
			goodMatches.fromList(goodMatchesList);

			Features2d.drawMatches(objectImage, objectKeyPoints, sceneImage,
					sceneKeyPoints, goodMatches, matchoutput, matchestColor,
					newKeypointColor, new MatOfByte(), 2);

			Highgui.imwrite("output//outputImage.jpg", outputImage);
			Highgui.imwrite("output//matchoutput.jpg", matchoutput);
			Highgui.imwrite("output//img.jpg", img);
		} else {
			System.out.println("Object Not Found");
		}

		System.out.println("Ended....");
	}
}