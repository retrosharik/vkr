package maps.convert.osm2gml;

import rescuecore2.misc.geometry.GeometryTools2D;
import rescuecore2.misc.geometry.Line2D;
import rescuecore2.misc.geometry.Point2D;
import rescuecore2.misc.geometry.Vector2D;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

/**
 * This step modify the map so that all shapes are traversable from their centroid.
 */
public class EnsureTraversabilityStep extends BaseModificationStep {
    private final double threshold;
    private static final int MAX_SPLIT_ITERATIONS = 5;

    public EnsureTraversabilityStep(TemporaryMap map) {
        super(map);
        threshold = ConvertTools.sizeOfMeters(map.getOSMMap(), 1);
    }


    @Override
    public String getDescription() {
        return "Ensure shapes are traversable";
    }

    @Override
    protected void step() {
        int totalSplits = 0;
        Collection<TemporaryObject> initialAllObjects = map.getAllObjects();
        List<TemporaryObject> modifiedShapes = new ArrayList<>();
        List<TemporaryObject> addedShapes = new ArrayList<>();

        setProgressLimit(initialAllObjects.size());

        for (TemporaryObject object : initialAllObjects) {

            // Check if centroid can reach passable edges.
            if (isValidObject(object)) continue;

            ArrayDeque<TemporaryObject> toProcess = new ArrayDeque<>();
            toProcess.offer(object);
            modifiedShapes.add(object);

            int iteration = 0;
            while (!toProcess.isEmpty() && iteration < MAX_SPLIT_ITERATIONS) {
                TemporaryObject current = toProcess.poll();
                List<TemporaryObject> newShapes = splitObject(current);
                totalSplits++;
                iteration++;

                for (TemporaryObject newObject : newShapes) {
                    if (isValidObject(newObject)) {
                        addedShapes.add(newObject);
                    } else {
                        toProcess.push(newObject);
                    }
                }
            }

            bumpProgress();
        }

        if (!modifiedShapes.isEmpty()) {
            for (TemporaryObject object : modifiedShapes) map.removeTemporaryObject(object);
            for (TemporaryObject object : addedShapes) map.addTemporaryObject(object);
            map.resynchronizeStateFromObjects();
        }

        setStatus("Split " + totalSplits + " objects.");
        visualizeDifference(modifiedShapes, addedShapes, "Splitting polygon");
    }

    private boolean isValidObject(TemporaryObject object) {
        Point2D centroid = object.getCentroid();

        // Skip shapes with fewer than 4 edges; triangles are always traversable.
        List<DirectedEdge> edges = object.getEdges();
        if (edges.size() < 4) return true;

        // Separate edges into passable and impassable.
        List<Line2D> passableLines = new ArrayList<>();
        List<Line2D> impassableLines = new ArrayList<>();
        for (DirectedEdge directedEdge : object.getEdges()) {
            int attachedCount = map.getAttachedObjects(directedEdge.getEdge()).size();
            if (1 < attachedCount) {
                passableLines.add(directedEdge.getLine());
            } else {
                impassableLines.add(directedEdge.getLine());
            }
        }

        for (Line2D line : passableLines) {
            Point2D edgeCentre = line.getPoint(0.5);
            Line2D lineOfSight = new Line2D(centroid, edgeCentre);
            boolean intersects = intersectsAny(lineOfSight, impassableLines);
            if (intersects) return false;
        }

        return true;
    }

    private List<TemporaryObject> splitObject(TemporaryObject object) {
        List<Point2D> concavePoints = findConcaveVertices(object.getVertices());
        List<Line2D> splitLines = generateSplitLines(object, concavePoints);
        if (splitLines.isEmpty()) return Collections.emptyList();

        double minConcaveCount = concavePoints.size();
        Line2D bestSplitLine = null;
        for (Line2D line : splitLines) {
            List<Point2D> polygon = object.getVertices();
            Point2D point1 = line.getOrigin();
            Point2D point2 = line.getEndPoint();
            List<List<Point2D>> splitPolygons = splitPolygon(polygon, point1, point2);
            if (splitPolygons.size() != 2) continue;

            double totalArea = GeometryTools2D.computeArea(polygon);
            double area1 = GeometryTools2D.computeArea(splitPolygons.get(0));
            double area2 = GeometryTools2D.computeArea(splitPolygons.get(1));

            // Skip this line if it extends outside the original polygon.
            if (threshold * threshold < area1 + area2 - totalArea) continue;

            int concave1 = findConcaveVertices(splitPolygons.get(0)).size();
            int concave2 = findConcaveVertices(splitPolygons.get(1)).size();
            int totalConcaveCount = concave1 + concave2;

            if (totalConcaveCount < minConcaveCount) {
                minConcaveCount = totalConcaveCount;
                bestSplitLine = line;
            }
        }

        if (bestSplitLine == null) return Collections.emptyList();

        Node startNode = findNode(object, bestSplitLine.getOrigin());
        Node endNode = findNode(object, bestSplitLine.getEndPoint());

        return splitObject(object, startNode, endNode);
    }

    private Node findNode(TemporaryObject object, Point2D point) {
        List<Node> nodes = object.getNodes();
        for (Node node : nodes) {
            if (node.getCoordinates().equals(point)) return node;
        }
        return null;
    }

    private List<TemporaryObject> splitObject(TemporaryObject object, Node start, Node end) {
        List<DirectedEdge> edges = object.getEdges();
        List<DirectedEdge> path1 = getEdgesPath(edges, start, end);
        List<DirectedEdge> path2 = getEdgesPath(edges, end, start);

        TemporaryObject object1 = createTemporaryObject(object, path1);
        TemporaryObject object2 = createTemporaryObject(object, path2);

        if (object1 == null || object2 == null) return Collections.emptyList();

        return List.of(object1, object2);
    }

    private List<DirectedEdge> getEdgesPath(List<DirectedEdge> edges, Node start, Node end) {
        List<DirectedEdge> pathEdges = new ArrayList<>();
        int n = edges.size();
        boolean started = false;

        for (int i = 0; i < n * 2; i++) {
            DirectedEdge edge = edges.get(i % n);
            if (edge.getStartNode().equals(start)) started = true;
            if (started) pathEdges.add(edge);
            if (edge.getEndNode().equals(end) && started) break;
        }

        // Close path.
        Edge closeEdge = map.getEdge(end, start);
        boolean isForward = closeEdge.getStart().equals(end);
        pathEdges.add(new DirectedEdge(closeEdge, isForward));

        return pathEdges;
    }

    private List<List<Point2D>> splitPolygon(List<Point2D> polygon, Point2D point1, Point2D point2) {
        List<Point2D> path1 = getVerticesPath(polygon, point1, point2);
        List<Point2D> path2 = getVerticesPath(polygon, point2, point1);

        return List.of(path1, path2);
    }

    private List<Point2D> getVerticesPath(List<Point2D> vertices, Point2D start, Point2D end) {
        List<Point2D> pathVertices = new ArrayList<>();
        int n = vertices.size();
        boolean started = false;

        for (int i = 0; i < n * 2; i++) {
            Point2D vertex = vertices.get(i % n);
            if (vertex.equals(start)) started = true;
            if (started) pathVertices.add(vertex);
            if (vertex.equals(end) && started) break;
        }
        pathVertices.add(start); // Close path.

        return pathVertices;
    }

    private List<Point2D> findConcaveVertices(List<Point2D> vertices) {
        List<Point2D> concaveVertices = new ArrayList<>();
        int n = vertices.size();
        boolean isCounterclockwise = GeometryTools2D.isCounterClockwise(vertices);

        for (int i = 0; i < n; i++) {
            Point2D prevVertex = vertices.get((i - 1 + n) % n);
            Point2D currVertex = vertices.get(i);
            Point2D nextVertex = vertices.get((i + 1) % n);

            Vector2D vector1 = currVertex.minus(prevVertex);
            Vector2D vector2 = nextVertex.minus(currVertex);
            double cross = vector1.cross(vector2);
            double squaredThreshold = threshold * threshold;
            if (isCounterclockwise && cross < -squaredThreshold || !isCounterclockwise && squaredThreshold < cross) {
                concaveVertices.add(currVertex);
            }
        }

        return concaveVertices;
    }

    private static List<Line2D> generateSplitLines(TemporaryObject object, List<Point2D> concaveVertices) {
        List<Line2D> splitLines = new ArrayList<>();
        List<Point2D> polygonVertices = object.getVertices();
        int n = polygonVertices.size();

        for (Point2D concaveVertex : concaveVertices) {
            int index = polygonVertices.indexOf(concaveVertex);
            int prev = (index - 1 + n) % n;
            int next = (index + 1) % n;

            for (int i = 0; i < polygonVertices.size(); i++) {
                if (i == index || i == prev || i == next) continue;

                Point2D candidateVertex = polygonVertices.get(i);
                Line2D line = new Line2D(concaveVertex, candidateVertex);
                Line2D reverseLine = new Line2D(candidateVertex, concaveVertex);

                if (splitLines.contains(line) || splitLines.contains(reverseLine)) continue;

                splitLines.add(line);
            }
        }
        return splitLines;
    }

    // Create a new TemporaryObject of the same type as the original, using provided edges.
    private TemporaryObject createTemporaryObject(TemporaryObject original, List<DirectedEdge> path) {
        if (path.isEmpty()) return null;

        List<DirectedEdge> edgesCopy = new ArrayList<>(path);

        if (original instanceof TemporaryRoad) {
            return new TemporaryRoad(edgesCopy);
        } else if (original instanceof TemporaryIntersection) {
            return new TemporaryIntersection(edgesCopy);
        } else if (original instanceof TemporaryBuilding building) {
            return new TemporaryBuilding(edgesCopy, building.getBuildingID());
        } else {
            return null;
        }
    }

    // Check if a given line intersects with any line in a collection.
    private boolean intersectsAny(Line2D line, Collection<Line2D> others) {
        for (Line2D other : others) {
            if (crosses(line, other)) return true;
        }
        return false;
    }

    // Check if two line segments properly cross each other (excluding touching at endpoints).
    private boolean crosses(Line2D line1, Line2D line2) {
        // Compute intersection parameters along each line
        double intersection1 = line1.getIntersection(line2);
        double intersection2 = line2.getIntersection(line1);

        // If lines are parallel, they do not have a valid intersection.
        if (Double.isNaN(intersection1) || Double.isNaN(intersection2)) return false;

        // Define a small threshold to avoid counting endpoints as crossings.
        boolean isInternal1 = threshold < intersection1 && intersection1 < (1.0 - threshold);
        boolean isInternal2 = threshold < intersection2 && intersection2 < (1.0 - threshold);

        return isInternal1 && isInternal2;
    }

}
