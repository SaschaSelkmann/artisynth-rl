package artisynth.core.rl;

import java.util.ArrayList;

public class RlState {
	private ArrayList<RlComponent> rlComponents;
	private ArrayList<RlProp> rlProps;
	private RlMuscleProps excitations;
	private RlMuscleProps muscleForces;
	private double time;
	private boolean terminated;
	private boolean truncated;

	public RlState() {
		time = 0.;
		rlComponents = new ArrayList<RlComponent>();
		rlProps = new ArrayList<RlProp>();
		excitations = new RlMuscleProps();
		muscleForces = new RlMuscleProps();
		terminated = false;
		truncated = false;
	}

	public boolean isTerminated() { return terminated; }
	public void setTerminated(boolean terminated) { this.terminated = terminated; }

	public boolean isTruncated() { return truncated; }
	public void setTruncated(boolean truncated) { this.truncated = truncated; }
	
	public void setTime(double t) { 
		this.time = t;
	}
	
	public double getTime() {
		return this.time;
	}

	public ArrayList<RlComponent> getRlComponents() {
		return rlComponents;
	}	
	
	public ArrayList<RlProp> getProps() {
		return rlProps;
	}

	public ArrayList<Double> getExcitations() {
		return excitations.getProps();
	}

	public void setExcitations(ArrayList<Double> exc) {
		excitations.setProps(exc);
	}

	public ArrayList<Double> getMuscleForces() {
		return muscleForces.getProps();
	}

	public void setMuscleForces(ArrayList<Double> f) {
		muscleForces.setProps(f);
	}

	public void addAllRlComponents(ArrayList<RlComponent> list) {
		rlComponents.addAll(list);
	}
	
	public void addAllProps(ArrayList<RlProp> list) {
		rlProps.addAll(list);
	}
	
	public void addProp(RlProp prop) {
		rlProps.add(prop);
	}

	public int numComponents() {
		return rlComponents.size();
	}

	public int size(Boolean withMuscleProps) {
		int size = 0;
		for (RlComponent rlComponent : rlComponents) {
			size += rlComponent.getPosition() != null ? rlComponent.getPosition().size() : 0;
			size += rlComponent.getOrientation() != null ? rlComponent.getOrientation().size() : 0;
			size += rlComponent.getAngularVelocity() != null ? rlComponent.getAngularVelocity().size() : 0;
			size += rlComponent.getVelocity() != null ? rlComponent.getVelocity().size() : 0;
		}

		if (withMuscleProps) {
			size += excitations.getProps().size();
			size += muscleForces.getProps().size();
		}
		return size;
	}

}