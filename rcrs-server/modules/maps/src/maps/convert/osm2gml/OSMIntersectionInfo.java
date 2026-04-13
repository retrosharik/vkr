package maps.convert.osm2gml;

import java.awt.geom.Rectangle2D;
import java.util.*;

import java.awt.geom.Area;
import java.awt.geom.Path2D;

import rescuecore2.misc.geometry.Line2D;
import rescuecore2.misc.geometry.Point2D;
import rescuecore2.misc.geometry.Vector2D;
import rescuecore2.misc.geometry.GeometryTools2D;

//import rescuecore2.misc.gui.ShapeDebugFrame;
//import java.awt.Color;

import maps.osm.OSMNode;

/**
   Information about an OSM intersection.
*/
public class OSMIntersectionInfo implements OSMShape {
    //    private static ShapeDebugFrame debug = new ShapeDebugFrame();

    private OSMNode centre;
    private List<RoadAspect> roads;
    private List<Point2D> vertices;
    private Area area;

    /**
       Create an IntersectionInfo.
       @param centre The OSMNode at the centre of the intersection.
    */
    public OSMIntersectionInfo(OSMNode centre) {
        this.centre = centre;
        roads = new ArrayList<RoadAspect>();
    }

    /**
       Add an incoming road.
       @param road The incoming road.
    */
    public void addRoadSegment(OSMRoadInfo road) {
        if (road.getFrom() == centre && road.getTo() == centre) {
            System.out.println("Degenerate road found");
        }
        else {
            roads.add(new RoadAspect(road, centre));
        }
    }

    /**
     * Clear the list of connected road segments.
     * Used before rebuilding the intersection's connections after a merge.
     */
    public void clearRoadSegments() {
        if (roads != null) {
            roads.clear();
        }
    }

    /**
       Process this intersection and determine the vertices and area it covers.
       @param sizeOf1m The size of 1m in latitude/longitude.
    */
    public void process(double sizeOf1m) {
        vertices = new ArrayList<>();

        if (roads == null || roads.isEmpty()) {
            area = null;
        } else if (roads.size() == 1) {
            processSingleRoad(sizeOf1m);
            area = null;
        } else {
            processRoads(sizeOf1m);
        }
    }

    /**
       Get the OSMNode at the centre of this intersection.
       @return The OSMNode at the centre.
    */
    public OSMNode getCentre() {
        return centre;
    }

    /**
     * Get the underlying OSMNode that represents the key point of this intersection.
     * @return The underlying OSMNode.
     */
    public OSMNode getUnderlyingNode() {
        return centre;
    }

    /**
     * Get the representative geometric location of this intersection.
     * If the intersection polygon has been processed, it returns the centroid of that polygon.
     * Otherwise, it returns the location of the central OSMNode.
     * @return The location as a Point2D.
     */
    public Point2D getLocation() {
        if (area != null && !area.isEmpty()) {
            Rectangle2D bounds = area.getBounds2D();
            return new Point2D(bounds.getCenterX(), bounds.getCenterY());
        }
        // As a fallback, use the location of the central node.
        return new Point2D(centre.getLongitude(), centre.getLatitude());
    }

    @Override
    public Area getArea() {
        return area;
    }

    @Override
    public List<Point2D> getVertices() {
        return vertices;
    }

    @Override
    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("IntersectionInfo (centre ");
        result.append(centre);
        result.append(") [");
        for (Iterator<Point2D> it = vertices.iterator(); it.hasNext();) {
            result.append(it.next().toString());
            if (it.hasNext()) {
                result.append(", ");
            }
        }
        result.append("]");
        if (area == null) {
            result.append(" (degenerate)");
        }
        return result.toString();
    }

    private void processRoads(double sizeOf1m) {
        // Sort incoming roads counterclockwise about the centre.
        Point2D centrePoint = new Point2D(centre.getLongitude(), centre.getLatitude());
        CounterClockwiseSort sort = new CounterClockwiseSort(centrePoint);
        roads.sort(sort);

        Map<RoadAspect, Point2D[]> roadMouths = new HashMap<>();
        for (RoadAspect road : roads) {
            roadMouths.put(road, calculateRoadMouth(road, centrePoint, sizeOf1m));
        }

        // Go through each pair of adjacent incoming roads and connect their mouths.
        Iterator<RoadAspect> it = roads.iterator();
        RoadAspect first = it.next();
        RoadAspect previous = first;
        while (it.hasNext()) {
            RoadAspect next = it.next();
            // Add the right corner of the previous road's mouth
            vertices.add(roadMouths.get(previous)[1]);
            // Add the left corner of the next road's mouth
            vertices.add(roadMouths.get(next)[0]);

            // Connect the two corners
            previous.setRightEnd(roadMouths.get(previous)[1]);
            next.setLeftEnd(roadMouths.get(next)[0]);

            previous = next;
        }
        // Connect the last road back to the first one
        vertices.add(roadMouths.get(previous)[1]);
        vertices.add(roadMouths.get(first)[0]);
        previous.setRightEnd(roadMouths.get(previous)[1]);
        first.setLeftEnd(roadMouths.get(first)[0]);

        // If there are multiple vertices then compute the area
        if (vertices.size() > 2) {
            Path2D.Double path = new Path2D.Double();
            Iterator<Point2D> ix = vertices.iterator();
            Point2D p = ix.next();
            path.moveTo(p.getX(), p.getY());
            while (ix.hasNext()) {
                p = ix.next();
                path.lineTo(p.getX(), p.getY());
            }
            path.closePath();
            area = new Area(path.createTransformedShape(null));
        }
        else {
            area = null;
        }
    }

    private Point2D[] calculateRoadMouth(RoadAspect road, Point2D centrePoint, double sizeOf1m) {
        OSMNode farNode = road.getFarNode();
        Point2D farPoint = new Point2D(farNode.getLongitude(), farNode.getLatitude());

        // roadVector points FROM the far node TO the centre point.
        Vector2D roadVector = centrePoint.minus(farPoint);

        // Calculate the total length of the road segment.
        double roadLength = roadVector.getLength();

        // Determine the distance of the "mouth" from the centre.
        double desiredMouthDistance = Constants.ROAD_WIDTH * sizeOf1m * 1.5;
        double actualMouthDistance = Math.min(desiredMouthDistance, roadLength * 0.45);

        // Calculated the mouth's centre point using the *actual* safe distance.
        Vector2D oppositeVector = roadVector.scale(-1);
        Point2D mouthCentre = centrePoint.plus(oppositeVector.normalised().scale(actualMouthDistance));

        // Calculate the left and right corners of the mouth.
        Vector2D roadNormal = roadVector.getNormal().normalised().scale(Constants.ROAD_WIDTH * sizeOf1m / 2.0);

        Point2D leftCorner = mouthCentre.plus(roadNormal);
        Point2D rightCorner = mouthCentre.plus(roadNormal.scale(-1));

        return new Point2D[]{leftCorner, rightCorner};
    }

    /**
       This "intersection" has a single incoming road. Set the incoming road's left and right edges.
    */
    private void processSingleRoad(double sizeOf1m) {
        Point2D centrePoint = new Point2D(centre.getLongitude(), centre.getLatitude());
        RoadAspect road = roads.iterator().next();
        OSMNode node = road.getFarNode();
        Point2D nodePoint = new Point2D(node.getLongitude(), node.getLatitude());
        Vector2D nodeVector = centrePoint.minus(nodePoint);
        Vector2D nodeNormal = nodeVector.getNormal().normalised().scale(-Constants.ROAD_WIDTH * sizeOf1m / 2);
        Vector2D nodeNormal2 = nodeNormal.scale(-1);
        Point2D start1Point = nodePoint.plus(nodeNormal);
        Point2D start2Point = nodePoint.plus(nodeNormal2);
        Line2D line1 = new Line2D(start1Point, nodeVector);
        Line2D line2 = new Line2D(start2Point, nodeVector);
        Point2D end1 = line1.getPoint(1);
        Point2D end2 = line2.getPoint(1);
        road.setRightEnd(end1);
        road.setLeftEnd(end2);

        /*
          List<ShapeDebugFrame.ShapeInfo> shapes = new ArrayList<ShapeDebugFrame.ShapeInfo>();
          shapes.add(new ShapeDebugFrame.Line2DShapeInfo(new Line2D(nodePoint, centrePoint), "Single road", Color.BLUE, false));
          shapes.add(new ShapeDebugFrame.Line2DShapeInfo(new Line2D(nodePoint, nodeNormal), "Offset 1", Color.YELLOW, false));
          shapes.add(new ShapeDebugFrame.Line2DShapeInfo(new Line2D(nodePoint, nodeNormal2), "Offset 2", Color.CYAN, false));
          shapes.add(new ShapeDebugFrame.Point2DShapeInfo(start1Point, "Left start", Color.BLUE, true));
          shapes.add(new ShapeDebugFrame.Line2DShapeInfo(line1, "Left edge", Color.BLUE, true));
          shapes.add(new ShapeDebugFrame.Point2DShapeInfo(start2Point, "Right start", Color.WHITE, true));
          shapes.add(new ShapeDebugFrame.Line2DShapeInfo(line2, "Right edge", Color.WHITE, true));
          shapes.add(new ShapeDebugFrame.Point2DShapeInfo(end1, "Endpoint 1", Color.ORANGE, true));
          shapes.add(new ShapeDebugFrame.Point2DShapeInfo(end2, "Endpoint 2", Color.PINK, true));
          debug.show(shapes);
        */
    }

    private static class RoadAspect {
        private boolean forward;
        private OSMRoadInfo road;

        RoadAspect(OSMRoadInfo road, OSMNode intersection) {
            this.road = road;
            forward = intersection == road.getTo();
        }

        OSMRoadInfo getRoad() {
            return road;
        }

        OSMNode getFarNode() {
            return forward ? road.getFrom() : road.getTo();
        }

        void setLeftEnd(Point2D p) {
            if (forward) {
                road.setToLeft(p);
            }
            else {
                road.setFromRight(p);
            }
        }

        void setRightEnd(Point2D p) {
            if (forward) {
                road.setToRight(p);
            }
            else {
                road.setFromLeft(p);
            }
        }
    }

    private static class CounterClockwiseSort implements Comparator<RoadAspect> {
        private Point2D centre;

        /**
           Construct a CounterClockwiseSort with a reference point.
           @param centre The reference point.
        */
        public CounterClockwiseSort(Point2D centre) {
            this.centre = centre;
        }

        @Override
        public int compare(RoadAspect first, RoadAspect second) {
            double d1 = score(first);
            double d2 = score(second);
            if (d1 < d2) {
                return 1;
            }
            else if (d1 > d2) {
                return -1;
            }
            else {
                return 0;
            }
        }

        /**
           Compute the score for a RoadAspect - the amount of clockwiseness from 12 o'clock.
           @param aspect The RoadAspect.
           @return The amount of clockwiseness. This will be in the range [0..4) with 0 representing 12 o'clock, 1 representing 3 o'clock and so on.
        */
        public double score(RoadAspect aspect) {
            OSMNode node = aspect.getFarNode();
            Point2D point = new Point2D(node.getLongitude(), node.getLatitude());
            Vector2D v = point.minus(centre);
            double sin = v.getX() / v.getLength();
            double cos = v.getY() / v.getLength();
            if (Double.isNaN(sin) || Double.isNaN(cos)) {
                System.out.println(v);
                System.out.println(v.getLength());
            }
            return convert(sin, cos);
        }

        // CHECKSTYLE:OFF:MagicNumber
        private double convert(double sin, double cos) {
            if (sin >= 0 && cos >= 0) {
                return sin;
            }
            if (sin >= 0 && cos < 0) {
                return 2 - sin;
            }
            if (sin < 0 && cos < 0) {
                return 2 - sin;
            }
            if (sin < 0 && cos >= 0) {
                return 4 + sin;
            }
            throw new IllegalArgumentException("This should be impossible! What's going on? sin=" + sin + ", cos=" + cos);
        }
        // CHECKSTYLE:ON:MagicNumber
    }
}
