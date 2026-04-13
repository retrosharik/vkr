package maps.convert.osm2gml;

import java.awt.geom.Area;
import java.util.*;
import java.util.List;

import static maps.convert.osm2gml.ConvertTools.areaToTemporaryPassableShapes;

/**
 * This step cleans up overlaps between buildings and roads by subtracting
 * the building shapes from the road shapes.
 */
public class CleanBuildingOverlapsStep extends BaseModificationStep {

    public CleanBuildingOverlapsStep(TemporaryMap map) {
        super(map);
    }


    @Override
    public String getDescription() {
        return "Cleaning overlaps between buildings and passable areas";
    }

    @Override
    protected void step() {
        Collection<TemporaryBuilding> buildings = map.getBuildings();
        Collection<TemporaryObject> initialPassableShapes = new ArrayList<>(map.getAllPassableShapes());

        if (buildings.isEmpty() || initialPassableShapes.isEmpty()) {
            setStatus("No buildings or passable-areas to process.");
            return;
        }

        int cleanedCount = 0;
        List<TemporaryObject> newObjects = new ArrayList<>();
        List<TemporaryObject> objectsToRemove = new ArrayList<>();

        for (TemporaryObject shape : initialPassableShapes) {
            Area roadArea = new Area(shape.getShape());
            boolean modified = false;
            for (TemporaryBuilding building : buildings) {
                // Quick check using bounds for performance
                if (!shape.getBounds().intersects(building.getBounds())) {
                    continue;
                }

                Area buildingArea = new Area(building.getShape());
                Area intersection = new Area(roadArea);
                intersection.intersect(buildingArea);

                if (!intersection.isEmpty()) {
                    // Subtract the building's shape from the road
                    roadArea.subtract(buildingArea);
                    modified = true;
                }
            }
            if (modified) {
                objectsToRemove.add(shape);
                newObjects.addAll(areaToTemporaryPassableShapes(roadArea, shape, map));
                cleanedCount++;
            }
        }

        // Apply the changes to the map
        for (TemporaryObject object : objectsToRemove) {
            map.removeTemporaryObject(object);
        }
        for (TemporaryObject object : newObjects) {
            map.addTemporaryObject(object);
        }

        // After making major geometric changes, we must resynchronize
        // the map's entire low-level state with the new high-level objects.
        map.resynchronizeStateFromObjects();

        setStatus("Cleaned " + cleanedCount + " passable areas that overlapped with buildings.");

        visualizeDifference(initialPassableShapes, map.getAllPassableShapes(), "Building/Passable-areas Overlap Cleanup");
    }
}
