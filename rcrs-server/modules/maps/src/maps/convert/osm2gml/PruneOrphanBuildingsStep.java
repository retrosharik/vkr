package maps.convert.osm2gml;

import java.util.*;

/**
 * This step removes any buildings that are not connected to the main road network.
 * Such buildings are considered "orphaned" and are invalid in the simulation.
 */
public class PruneOrphanBuildingsStep extends BaseModificationStep {

    public PruneOrphanBuildingsStep(TemporaryMap map) {
        super(map);
    }

    @Override
    public String getDescription() {
        return "Pruning orphan buildings";
    }

    @Override
    protected void step() {
        Collection<TemporaryBuilding> initialBuildings = new ArrayList<>(map.getBuildings());

        if (initialBuildings.isEmpty()) {
            setStatus("No buildings to prune.");
            return;
        }

        Map<TemporaryBuilding, Set<TemporaryBuilding>> buildingAdjacency = buildBuildingAdjacency(initialBuildings);

        // Create a definitive set of all edges that belong to the road network.
        Set<Edge> roadNetworkEdges = new HashSet<>();
        for (TemporaryObject passable : map.getAllPassableShapes()) {
            for(DirectedEdge de : passable.getEdges()) {
                roadNetworkEdges.add(de.getEdge());
            }
        }

        // Find all buildings that are DIRECTLY connected to the road network.
        Queue<TemporaryBuilding> queue = new ArrayDeque<>();
        Set<TemporaryBuilding> reachableBuildings = new HashSet<>();

        for (TemporaryBuilding building : initialBuildings) {
            for (DirectedEdge de : building.getEdges()) {
                if (roadNetworkEdges.contains(de.getEdge())) {
                    if (reachableBuildings.add(building)) {
                        queue.add(building);
                    }
                    break;
                }
            }
        }

        while (!queue.isEmpty()) {
            TemporaryBuilding current = queue.poll();
            for (TemporaryBuilding neighbour : buildingAdjacency.get(current)) {
                if (reachableBuildings.add(neighbour)) {
                    queue.add(neighbour);
                }
            }
        }

        List<TemporaryBuilding> buildingsToRemove = new ArrayList<>(initialBuildings);
        buildingsToRemove.removeAll(reachableBuildings);

        for (TemporaryBuilding building : buildingsToRemove) {
            map.removeBuilding(building);
        }
        map.resynchronizeStateFromObjects();

        visualizeDifference(initialBuildings, map.getBuildings(), "Orphaned Building Pruning Results");
    }

    private Map<TemporaryBuilding, Set<TemporaryBuilding>> buildBuildingAdjacency(Collection<TemporaryBuilding> buildings) {
        Map<TemporaryBuilding, Set<TemporaryBuilding>> adjacency = new HashMap<>();
        for (TemporaryBuilding b : buildings) adjacency.put(b, new HashSet<>());

        List<TemporaryBuilding> buildingList = new ArrayList<>(buildings);
        for (int i = 0; i < buildingList.size(); i++) {
            for (int j = i + 1; j < buildingList.size(); j++) {
                TemporaryBuilding b1 = buildingList.get(i);
                TemporaryBuilding b2 = buildingList.get(j);
                if (areBuildingsConnected(b1, b2)) {
                    adjacency.get(b1).add(b2);
                    adjacency.get(b2).add(b1);
                }
            }
        }
        return adjacency;
    }

    private boolean areBuildingsConnected(TemporaryBuilding b1, TemporaryBuilding b2) {
        for (DirectedEdge de : b1.getEdges()) {
            Collection<TemporaryObject> attached = map.getAttachedObjects(de.getEdge());
            if (attached.contains(b2)) return true;
        }
        return false;
    }

}
