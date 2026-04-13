package maps.convert.osm2gml;

import rescuecore2.misc.geometry.GeometryTools2D;
import rescuecore2.misc.geometry.Line2D;
import rescuecore2.misc.geometry.Point2D;
import rescuecore2.misc.geometry.Vector2D;

import java.awt.geom.Area;
import java.util.*;

public class ConnectBuildingsStep extends BaseModificationStep {

    private final double maxConnectDistance;
    private final double minConnectDistance;
    private final double maxAngleDeviation;
    private final double entranceWidth;

    private record EntrancePlan(
            TemporaryIntersection entranceObject,
            Edge buildingEdge, Edge roadEdge,
            Node buildingNode1, Node buildingNode2,
            Node roadNode1, Node roadNode2
    ) {}

    public ConnectBuildingsStep(TemporaryMap map) {
        super(map);
        maxConnectDistance = ConvertTools.sizeOfMeters(map.getOSMMap(), 20);
        minConnectDistance = ConvertTools.sizeOfMeters(map.getOSMMap(), 1); // Nearby threshold
        maxAngleDeviation = 45;
        entranceWidth = ConvertTools.sizeOfMeters(map.getOSMMap(), Constants.ROAD_WIDTH);
    }

    @Override
    public String getDescription() {
        return "Connecting buildings to roads";
    }

    @Override
    protected void step() {
        List<TemporaryBuilding> buildings = new ArrayList<>(map.getBuildings());
        List<TemporaryIntersection> entrances = new ArrayList<>();
        setProgressLimit(buildings.size());

        SpatialGrid<TemporaryObject> roadGrid = new SpatialGrid<>(map.getBounds(), maxConnectDistance * 2);
        for (TemporaryRoad road : map.getRoads()) {
            roadGrid.add(road);
        }


        for (int i = 0; i < buildings.size(); i++) {
            TemporaryBuilding building = buildings.get(i);
            setProgress(i);

            if (isAlreadyConnected(building, map.getRoads())) continue;

            EntrancePlan bestPlan = findBestPlanForBuilding(building, roadGrid);
            if (bestPlan != null) {
                map.splitEdge(bestPlan.buildingEdge(), bestPlan.buildingNode1(), bestPlan.buildingNode2());
                map.splitEdge(bestPlan.roadEdge(), bestPlan.roadNode1(), bestPlan.roadNode2());
                map.addIntersection(bestPlan.entranceObject());

                entrances.add(bestPlan.entranceObject());
            }
        }

        if (!entrances.isEmpty()) {
            map.resynchronizeStateFromObjects();
        }

        setProgress(buildings.size());
        setStatus("Created " + entrances.size() + " new entrances for buildings.");
        visualizeDifference(Collections.emptyList(), entrances, "Building Connection Results");
    }

    private EntrancePlan findBestPlanForBuilding(TemporaryBuilding building, SpatialGrid<TemporaryObject> roadGrid) {
        EntrancePlan bestPlan = null;
        double bestAngleDeviation = Double.MAX_VALUE;

        for (DirectedEdge de : building.getEdges()) {
            Edge buildingEdge = de.getEdge();
            if (buildingEdge.getLine().getDirection().getLength() < entranceWidth) continue;

            for (TemporaryObject obj : roadGrid.getNearbyItems(building)) {
                TemporaryRoad road = (TemporaryRoad) obj;
                for (DirectedEdge roadDE : road.getEdges()) {
                    Edge roadEdge = roadDE.getEdge();

                    boolean isPassableEdge = 1 < map.getAttachedObjects(roadEdge).size();
                    if (isPassableEdge) continue;

                    boolean isRoadEdgeTooShort = roadEdge.getLine().getDirection().getLength() < entranceWidth;
                    if (isRoadEdgeTooShort) continue;

                    Point2D wallMidPoint = buildingEdge.getMidPoint();
                    Line2D roadLine = roadEdge.getLine();
                    Point2D connectingPoint = GeometryTools2D.getClosestPointOnSegment(roadLine, wallMidPoint);

                    // Safely calculate the entrance roof on the road, sliding if necessary.
                    double distFromStart = GeometryTools2D.getDistance(roadLine.getOrigin(), connectingPoint);
                    double distFromEnd = roadLine.getDirection().getLength() - distFromStart;
                    double halfWidth = entranceWidth / 2.0;
                    if (distFromStart < halfWidth) {
                        double slideAmount = halfWidth - distFromStart;
                        connectingPoint = connectingPoint.plus(roadLine.getDirection().normalised().scale(slideAmount));
                    } else if (distFromEnd < halfWidth) {
                        double slideAmount = halfWidth - distFromEnd;
                        connectingPoint = connectingPoint.plus(roadLine.getDirection().normalised().scale(-slideAmount));
                    }

                    // Define the entrance base on the building wall.
                    Vector2D wallVector = buildingEdge.getLine().getDirection().normalised();
                    Node b1 = map.getNode(wallMidPoint.plus(wallVector.scale(-halfWidth)));
                    Node b2 = map.getNode(wallMidPoint.plus(wallVector.scale(halfWidth)));

                    // Define the entrance top using these safe distances.
                    Vector2D roadVector = roadLine.getDirection().normalised();
                    Node r1 = map.getNode(connectingPoint.plus(roadVector.scale(-halfWidth)));
                    Node r2 = map.getNode(connectingPoint.plus(roadVector.scale(halfWidth)));

                    // Create entrance shape
                    List<DirectedEdge> entranceEdges = new ArrayList<>();
                    if (0 < wallVector.dot(roadVector)) {
                        entranceEdges.add(map.getDirectedEdge(b1, b2));
                        entranceEdges.add(map.getDirectedEdge(b2, r2));
                        entranceEdges.add(map.getDirectedEdge(r2, r1));
                        entranceEdges.add(map.getDirectedEdge(r1, b1));
                    } else {
                        entranceEdges.add(map.getDirectedEdge(b1, b2));
                        entranceEdges.add(map.getDirectedEdge(b2, r1));
                        entranceEdges.add(map.getDirectedEdge(r1, r2));
                        entranceEdges.add(map.getDirectedEdge(r2, b1));
                    }

                    TemporaryIntersection entrance = new TemporaryIntersection(entranceEdges);

                    Line2D entranceCenterLine = new Line2D(wallMidPoint, connectingPoint);
                    double entranceLength = entranceCenterLine.getDirection().getLength();

                    // This prevents creating entrances that are too short to be meaningful
                    // or are likely to cause geometric instability.
                    boolean isConnectDistanceTooShort = entranceLength < minConnectDistance;
                    boolean isConnectDistanceTooLong = maxConnectDistance < entranceLength;
                    if (isConnectDistanceTooShort || isConnectDistanceTooLong) continue;

                    double angleDeviation = calculateAngleDeviation(entranceCenterLine, buildingEdge, roadEdge);
                    boolean exceedsAngleToTolerance = maxAngleDeviation < angleDeviation;
                    if (exceedsAngleToTolerance) continue;

                    if (hasCollision(entrance, building, road)) continue;

                    if (angleDeviation < bestAngleDeviation) {
                        bestAngleDeviation = angleDeviation;
                        bestPlan = new EntrancePlan(entrance, buildingEdge, roadEdge, b1, b2, r1, r2);
                    }
                }
            }
        }

        return bestPlan;
    }

    private double calculateAngleDeviation(Line2D centerLine, Edge buildingEdge, Edge roadEdge) {
        double angleToBuilding = Math.abs(90.0 - GeometryTools2D.getAngleBetweenVectors(
                centerLine.getDirection(), buildingEdge.getLine().getDirection()));
        double angleToRoad = Math.abs(90.0 - GeometryTools2D.getAngleBetweenVectors(
                centerLine.getDirection(), roadEdge.getLine().getDirection()));
        return Math.max(angleToBuilding, angleToRoad);
    }

    private boolean isAlreadyConnected(TemporaryBuilding building, Collection<TemporaryRoad> roads) {
        Set<Edge> buildingEdges = new HashSet<>();
        for (DirectedEdge de : building.getEdges()) {
            buildingEdges.add(de.getEdge());
        }
        for (TemporaryRoad road : roads) {
            for (DirectedEdge de : road.getEdges()) {
                if (buildingEdges.contains(de.getEdge())) {
                    return true;
                }
            }
        }

        return false;
    }

    private boolean hasCollision(TemporaryIntersection candidate, TemporaryBuilding building, TemporaryRoad road) {
        if (candidate.getShape() == null) return false;
        Area entranceArea = new Area(candidate.getShape());

        for (TemporaryObject otherObject : map.getAllObjects()) {
            if (otherObject.getShape() == null) continue;

            Area otherArea = new Area(otherObject.getShape());
            otherArea.intersect(entranceArea);

            if (otherArea.isEmpty()) continue;
            if (!otherObject.equals(building) && !otherObject.equals(road)) return true;

            // A significant collision was found
            if (isSignificantOverlap(otherArea, map)) return true;
        }

        return false;
    }

    private boolean isSignificantOverlap(Area intersectionArea, TemporaryMap map) {
        double oneMeterInDegrees = ConvertTools.sizeOf1Metre(map.getOSMMap());
        double epsilon = Math.pow(oneMeterInDegrees / 10, 2);
        double geometricArea = ConvertTools.getGeometricArea(intersectionArea);
        return epsilon < geometricArea;
    }
}
