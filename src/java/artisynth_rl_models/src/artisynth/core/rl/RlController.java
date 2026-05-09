/**
 * Copyright (c) 2019, by the Authors: Amir Abdi (UBC)
 * Based on the TrackingController by Ian Stavness (UBC)
 *
 * This software is freely available under a 2-clause BSD license. Please see
 * the LICENSE file in the ArtiSynth distribution directory for details.
 */
package artisynth.core.rl;

import java.util.concurrent.locks.*;
import java.util.LinkedHashMap;
import java.util.Map;
import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JSeparator;

import artisynth.core.driver.Main;
import artisynth.core.driver.Scheduler;
import artisynth.core.moviemaker.MovieMaker;
import artisynth.core.gui.ControlPanel;
import artisynth.core.inverse.TargetFrame;
import artisynth.core.inverse.TargetPoint;
import artisynth.core.mechmodels.ExcitationComponent;
import artisynth.core.mechmodels.Frame;
import artisynth.core.mechmodels.MechSystemBase;
import artisynth.core.mechmodels.MechSystemModel;
import artisynth.core.mechmodels.MotionTargetComponent;
import artisynth.core.mechmodels.MultiPointMuscle;
import artisynth.core.mechmodels.Muscle;
import artisynth.core.mechmodels.MuscleExciter;
import artisynth.core.mechmodels.Point;
import artisynth.core.mechmodels.PointList;
import artisynth.core.mechmodels.RigidBody;
import artisynth.core.mechmodels.MotionTarget.TargetActivity;
import artisynth.core.modelbase.ComponentChangeEvent;
import artisynth.core.modelbase.ComponentList;
import artisynth.core.modelbase.ComponentListImpl;
import artisynth.core.modelbase.ComponentState;
import artisynth.core.modelbase.ComponentUtils;
import artisynth.core.modelbase.CompositeComponent;
import artisynth.core.modelbase.ControllerBase;
import artisynth.core.modelbase.ModelComponent;
import artisynth.core.modelbase.ReferenceList;
import artisynth.core.modelbase.RenderableComponent;
import artisynth.core.modelbase.RenderableComponentList;
import maspack.geometry.PolygonalMesh;
import maspack.matrix.VectorNd;
import maspack.properties.PropertyList;
import maspack.render.RenderList;
import maspack.render.RenderProps;
import maspack.render.Renderable;
import maspack.render.Renderer.FaceStyle;
import maspack.render.Renderer.PointStyle;
import maspack.util.ReaderTokenizer;

public class RlController extends ControllerBase
		implements CompositeComponent, RenderableComponent, RlControllerInterface {

	RlModelInterface myInverseModel;
	protected ComponentListImpl<ModelComponent> myComponents;

	// list of target points that store the location of motion targets
	protected PointList<TargetPoint> targetPoints;

	// list of target frames that store the location and rotation of targets
	protected RenderableComponentList<TargetFrame> targetFrames;

	// reference lists to original points and frames
	protected ReferenceList sourcePoints;
	protected ReferenceList sourceFrames;

	protected ArrayList<MotionTargetComponent> mySources;
	protected ArrayList<MotionTargetComponent> myTargets;

	// list of all muscle exciters
	protected ComponentList<ExcitationComponent> exciters;
	protected ArrayList<Double> excitationValues = new ArrayList<Double>();	
	protected RlState nextState = new RlState();
	protected double lastStateT = 0.0;

	RlRestApi networkHandler;
	protected MechSystemBase myMech;
	private String name;

	protected RenderProps targetRenderProps;
	protected RenderProps sourceRenderProps;

	protected Boolean excitersUpToDate = true;
	protected Boolean nextStateUpToDate = false;
	protected Boolean getNextState = false;
	protected Boolean nextStateUntilConvergence = false;

	// Pluggable advance policy. Default is deterministic stepped advance —
	// see StepStrategy. Demos can override via setStepStrategy().
	private StepStrategy stepStrategy = new SteppedStrategy(5);

	// Reset coordination: HTTP thread sets flags, simulation thread executes reset in apply()
	private volatile boolean resetPending = false;
	private volatile boolean resetZeroExcitations = false;
	private volatile RlState resetResult = null;
	private final Object resetLock = new Object();

	protected boolean targetsVisible = true;
	protected boolean sourcesVisible = true;
	protected boolean enabled = true;
	protected boolean debug = false;
	protected boolean testTrial = false;
	
	protected double targetsPointRadius = DEFAULT_TARGET_RADIUS;
	protected int targetsLineWidth = DEFAULT_TARGET_LINE_WIDTH;
	public static final boolean DEFAULT_DEBUG = true;
	private static final double DEFAULT_TARGET_RADIUS = 0.4d;
	private static final int DEFAULT_TARGET_LINE_WIDTH = 2;
	
	public Random random = new Random();

	/**
	 * Set the Rl model which implements the RlModelInterface interface.
	 */
	public void setInverseModel(RlModelInterface model) {
		myInverseModel = model;
	}

	public RlController(MechSystemBase m, RlModelInterface model, String name, int port, Boolean getNextState) {
		this(m, model, name, port);
		this.getNextState = getNextState;
	}

	public RlController(MechSystemBase m, RlModelInterface model, String name, int port) {
		super();
		setMech(m);
		setName(name);
		setInverseModel(model);

		networkHandler = new RlRestApi(this, port);

		myComponents = new ComponentListImpl<ModelComponent>(ModelComponent.class, this);
		mySources = new ArrayList<MotionTargetComponent>();
		myTargets = new ArrayList<MotionTargetComponent>();

		// setup points
		targetPoints = new PointList<TargetPoint>(TargetPoint.class, "targetPoints");
		// always show this component, even if it's empty
		targetPoints.setNavpanelVisibility(ModelComponent.NavpanelVisibility.ALWAYS);
		add(targetPoints);
		sourcePoints = new ReferenceList("sourcePoints");
		add(sourcePoints);

		// setup frames
		targetFrames = new RenderableComponentList<TargetFrame>(TargetFrame.class, "targetFrames");
		// always show this component, even if it's empty:
		targetFrames.setNavpanelVisibility(ModelComponent.NavpanelVisibility.ALWAYS);
		add(targetFrames);
		sourceFrames = new ReferenceList("sourceFrames");
		add(sourceFrames);

		exciters = new ComponentList<ExcitationComponent>(ExcitationComponent.class, "excitationSources");
		// always show this component, even if it's empty:
		exciters.setNavpanelVisibility(ModelComponent.NavpanelVisibility.ALWAYS);
		add(exciters);

		initTargetRenderProps();
		initSourceRenderProps();
	}

	public synchronized RlState getNextState(double t0, double t1) {
		double currentStep = t0;
		MechSystemModel mySys = (MechSystemModel) myMech;
		ComponentState saveState = mySys.createState(null);
		mySys.getState(saveState);

		if (nextStateUntilConvergence) {
			VectorNd prev_f = new VectorNd();
			VectorNd f = new VectorNd();
			mySys.getActiveForces(prev_f);

			double thres = 0.1;
			double force_diffs_norm = 1000;
			int iter = 0, max_iters = 10;
			double stepSize = t1 - t0;

			// repeat until convergence
			while (force_diffs_norm > thres && iter < max_iters) {
				mySys.preadvance(currentStep, currentStep + stepSize, 0);
				mySys.advance(currentStep, currentStep + stepSize, 0);

				mySys.getActiveForces(f);
				VectorNd diff = f.copyAndSub(prev_f);
				force_diffs_norm = diff.norm(); // faster than norm?

				prev_f = f.copy();
				iter++;
				currentStep += stepSize;
				Log.debug(iter + " * " + force_diffs_norm);
			}
			Log.debug("Next state converged");
		} else {
			double stepSize;
			if (lastStateT == 0.0)
				stepSize = t1 - t0;
			else
				stepSize = t0 - lastStateT;

			mySys.preadvance(currentStep, currentStep + stepSize, 0);
			mySys.advance(currentStep, currentStep + stepSize, 0);
			lastStateT = t0;
		}

		// Create the RlState and fill with values
		RlState rlState = new RlState();
		rlState.addAllRlComponents(getRlComponents(getSources()));
		rlState.addAllRlComponents(getRlComponents(getTargets()));
		rlState.setExcitations(this.getExcitations());

		// reset state
		mySys.setState(saveState);

		// inform about the next state being ready to send to the agent
		nextStateUpToDate = true;
		notify();

		return rlState;
	}

	@Override
	public void prerender(RenderList list) {
		super.prerender(list);
		recursivelyPrerender(this, list);
	}

	protected void recursivelyPrerender(CompositeComponent comp, RenderList list) {

		for (int i = 0; i < comp.numComponents(); i++) {
			ModelComponent c = comp.get(i);
			if (c instanceof Renderable) {
				list.addIfVisible((Renderable) c);
			} else if (c instanceof CompositeComponent) {
				recursivelyPrerender((CompositeComponent) c, list);
			}
		}
	}

	public void setMech(MechSystemBase m) {
		//setModel(m); John found this unnecessary
		myMech = m;
	}

	protected void add(ModelComponent comp) {
		myComponents.add(comp);
	}

	public MotionTargetComponent addMotionTarget(MotionTargetComponent source) {
		mySources.add(source);
		MotionTargetComponent target = null;
		if (source instanceof Point) {
			target = addTargetPoint((Point) source);
		} else if (source instanceof Frame) {
			target = addTargetFrame((RigidBody) source);
		} else
			throw new UnsupportedOperationException("not implemented");
		return target;
	}

	private TargetPoint addTargetPoint(Point source) {
		TargetPoint tpnt = new TargetPoint();
		tpnt.setName((source.getName() != null ? source.getName() : String.format("p%d", source.getNumber())) + "_ref");
		tpnt.setState(source);
		tpnt.setTargetActivity(TargetActivity.PositionVelocity);
		myTargets.add(tpnt);

		targetPoints.add(tpnt);

		return tpnt;
	}

	private TargetFrame addTargetFrame(RigidBody source) {
		TargetFrame tframe = new TargetFrame();
		tframe.setPose(source.getPose());
		tframe.setName(
				(source.getName() != null ? source.getName() : String.format("rb%d", source.getNumber())) + "_ref");

		tframe.setState(source);
		tframe.setTargetActivity(TargetActivity.PositionVelocity);

		myTargets.add(tframe);

		// add mesh to TargetFrame
		PolygonalMesh mesh = null;
		if ((mesh = source.getSurfaceMesh()) != null) {
			tframe.setSurfaceMesh(mesh.clone(), source.getSurfaceMeshComp().getFileName());
			tframe.setRenderProps(source.getRenderProps());
			RenderProps.setDrawEdges(tframe, true);
			RenderProps.setFaceStyle(tframe, FaceStyle.NONE);
		}

		targetFrames.add(tframe);
		return tframe;
	}

	public void addExciter(ExcitationComponent ex) {
		exciters.add(ex);
	}

	/**
	 * Returns the set of targets
	 */
	public ArrayList<MotionTargetComponent> getMotionTargets() {
		return myTargets;
	}

	public void setTargetRenderProps(RenderProps rend) {
		targetRenderProps.set(rend);

		targetPoints.setRenderProps(targetRenderProps);
		targetFrames.setRenderProps(targetRenderProps);
	}

	public void setSourceRenderProps(RenderProps rend) {
		sourceRenderProps.set(rend);

		for (MotionTargetComponent p : mySources) {
			if (p instanceof Point) {
				((Point) p).setRenderProps(sourceRenderProps);
			} else if (p instanceof Frame) {
				((Frame) p).setRenderProps(sourceRenderProps);
			}
		}
	}

	public void initTargetRenderProps() {
		targetRenderProps = new RenderProps();
		targetRenderProps.setDrawEdges(true);
		targetRenderProps.setFaceStyle(FaceStyle.NONE);
		targetRenderProps.setLineColor(Color.CYAN);
		targetRenderProps.setLineWidth(2);
		targetRenderProps.setPointColor(Color.CYAN);
		targetRenderProps.setPointStyle(PointStyle.SPHERE);
		// set target point radius explicitly
		targetRenderProps.setPointRadius(0.3);
		targetRenderProps.setVisible(true);
		targetRenderProps.setPointRadius(targetsPointRadius);

		targetPoints.setRenderProps(targetRenderProps);
		targetFrames.setRenderProps(targetRenderProps);
	}

	public void initSourceRenderProps() {
		sourceRenderProps = new RenderProps();
		sourceRenderProps.setDrawEdges(true);
		sourceRenderProps.setFaceStyle(FaceStyle.NONE);
		sourceRenderProps.setLineColor(Color.CYAN);
		sourceRenderProps.setLineWidth(2);
		sourceRenderProps.setPointColor(Color.CYAN);
		sourceRenderProps.setPointStyle(PointStyle.SPHERE);
		sourceRenderProps.setVisible(true);
		;
		// modRenderProps.setAlpha(0.5);

		setSourceRenderProps(sourceRenderProps);
	}

	public ArrayList<MotionTargetComponent> getTargets() {
		return myTargets;
	}

	public ArrayList<MotionTargetComponent> getSources() {
		return mySources;
	}

	@Override
	public String getName() {
		return name;
	}

	@Override
	public void setName(String name) throws IllegalArgumentException {
		this.name = name;
	}

	@Override
	public boolean hasState() {
		return false;
	}

	@Override
	public void scan(ReaderTokenizer rtok, Object ref) throws IOException {
		super.scan(rtok, ref);
	}

	@Override
	public ModelComponent get(String nameOrNumber) {
		return myComponents.get(nameOrNumber);
	}

	@Override
	public ModelComponent get(int idx) {
		return myComponents.get(idx);
	}

	@Override
	public ModelComponent getByNumber(int num) {
		return myComponents.getByNumber(num);
	}

	@Override
	public int numComponents() {
		return myComponents.size();
	}

	@Override
	public int indexOf(ModelComponent comp) {
		return myComponents.indexOf(comp);
	}

	@Override
	public ModelComponent findComponent(String path) {
		return ComponentUtils.findComponent(this, path);
	}

	@Override
	public int getNumberLimit() {
		return myComponents.getNumberLimit();
	}

	@Override
	public NavpanelDisplay getNavpanelDisplay() {
		return CompositeComponent.NavpanelDisplay.NORMAL;
	}

	@Override
	public void componentChanged(ComponentChangeEvent e) {
		myComponents.componentChanged(e);
		notifyParentOfChange(e);
	}

	@Override
	public void updateNameMap(String newName, String oldName, ModelComponent comp) {
		myComponents.updateNameMap(newName, oldName, comp);
	}

	@Override
	public boolean hierarchyContainsReferences() {
		return false;
	}

	@Override
	public PropertyList getAllPropertyInfo() {
		return myProps;
	}

	public static PropertyList myProps = new PropertyList(RlController.class, ControllerBase.class);
	static {
		myProps.add("renderProps * *", "render properties", null);
		myProps.add("enabled isEnabled *", "enable/disable controller", true);
		myProps.add("targetsVisible * *", "allow showing or hiding of targets markers", true);
		myProps.add("sourcesVisible * *", "allow showing or hiding of source markers", true);
		myProps.add("targetsPointRadius * *", "set size of target markers", DEFAULT_TARGET_RADIUS);
		myProps.add("targetsLineWidth * *", "set width of target lines", DEFAULT_TARGET_LINE_WIDTH);
		myProps.add("debug isDebug", "enables output of debug info to the console", DEFAULT_DEBUG);
	}

	public ControlPanel getRlControlPanel() {
		ControlPanel cp = new ControlPanel("RlControlPanel");

		cp.addWidget("Enabled", this, "enabled");

		cp.addWidget(new JSeparator());
		cp.addWidget(new JLabel("Render"));
		cp.addWidget("Sources", this, "sourcesVisible");

		cp.addWidget("Target", this, "targetsVisible");
		cp.addWidget("Targets Point Radius", this, "targetsPointRadius");
		cp.addWidget("Targets Line Width", this, "targetsLineWidth");

		cp.addWidget(new JSeparator());
		cp.addWidget(new JLabel("Control"));
		JButton resetStateButton = new JButton("Reset State");
		resetStateButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				resetState(false);
			}
		});

		JButton randomActionButton = new JButton("Random Action");
		randomActionButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				setRandomExcitations();
			}
		});

		javax.swing.JPanel panel = new JPanel();
		panel.add(resetStateButton);
		panel.add(randomActionButton);
		cp.addWidget(panel);

		cp.addWidget(new JSeparator());
		cp.addWidget("Debug", this, "debug");

		return cp;
	}

	public ControlPanel getMuscleControlPanel() {
		ControlPanel cp = new ControlPanel("MuscleControlPanel");

		for (ExcitationComponent ex : exciters) {
			cp.addWidget(ex.getName(), ex, "excitation");
		}

		return cp;
	}

	/**
	 * Show or hide the sources
	 */
	public void setSourcesVisible(boolean show) {
		ArrayList<MotionTargetComponent> moTargetParticles = mySources;
		for (MotionTargetComponent p : moTargetParticles) {
			if (p instanceof RenderableComponent) {
				RenderProps.setVisible((RenderableComponent) p, show);
			}
		}
		sourcesVisible = show;
	}

	public void setTargetsPointRadius(double radius) {
		targetsPointRadius = radius;
		RenderProps.setPointRadius(targetPoints, radius);
	}

	public double getTargetsPointRadius() {
		return targetsPointRadius;
	}

	public void setTargetsLineWidth(int width) {
		targetsLineWidth = width;
		RenderProps.setLineWidth(targetFrames, width);
	}

	public int getTargetsLineWidth() {
		return targetsLineWidth;
	}

	public boolean getSourcesVisible() {
		return sourcesVisible;
	}

	public boolean getTargetsVisible() {
		return targetsVisible;
	}

	public void setTargetsVisible(boolean visible) {
		ArrayList<MotionTargetComponent> moTargetParticles = getTargets();
		for (MotionTargetComponent p : moTargetParticles) {
			if (p instanceof RenderableComponent) {
				RenderProps.setVisible((RenderableComponent) p, visible);
			}
		}
		targetsVisible = visible;
	}

	public boolean isDebug() {
		return debug;
	}

	public void setDebug(boolean debug) {
		this.debug = debug;
		Log.DEBUG = this.debug;
	}

	public boolean isEnabled() {
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	@Override
	public void initialize(double t0) {
		super.initialize(t0);
		Thread.currentThread().setPriority(Thread.MAX_PRIORITY);
	}

	@Override
	public void apply(double t0, double t1) {
		if (resetPending) {
			myInverseModel.resetState();
			if (resetZeroExcitations) {
				setExcitationsZero();
			}
			resetResult = getState();
			synchronized (resetLock) {
				resetPending = false;
				resetLock.notifyAll();
			}
			return;
		}

		if (!excitersUpToDate) {

			for (int i = 0; i < excitationValues.size(); ++i) {
				exciters.get(i).setExcitation(excitationValues.get(i));
			}
			excitersUpToDate = true;

			if (getNextState) {
				nextState = getNextState(t0, t1);
				Log.debug("nextStateUpToDate " + nextStateUpToDate);
			}
		}
	}

	// --------------- Implement RlControllerInterface ---------
	public int getObservationSize() {
		return getState().size(false);
	}

	public int getStateSize() {
		return getState().size(true);
	}

	public int getActionSize() {
		return getExcitations().size();
	}

	public RlState getState() {
		// real current position
		ArrayList<MotionTargetComponent> sources = getSources();

		// destination, ref, target
		ArrayList<MotionTargetComponent> targets = getTargets();

		// results
		RlState rlState = new RlState();
		rlState.addAllRlComponents(getRlComponents(sources));
		rlState.addAllRlComponents(getRlComponents(targets));

		rlState.setExcitations(getExcitations());
		rlState.setMuscleForces(getMuscleForces());
		
		rlState.addAllProps(myInverseModel.getRlProps());
		
		rlState.setTime(myInverseModel.getTime());

		Log.debug("Get State state.size = " + rlState.numComponents());
		return rlState;
	}

	private ArrayList<RlComponent> getRlComponents(ArrayList<MotionTargetComponent> comps) {
		ArrayList<RlComponent> state = new ArrayList<RlComponent>(comps.size());
		for (MotionTargetComponent component : comps) {
			if (component instanceof Point) {
				RlComponent rlComponent = new RlComponent();
				double[] values = new double[3];

				rlComponent.setName(component.getName());

				((Point) component).getVelocity().get(values);
				rlComponent.setVelocity(values);

				((Point) component).getPosition().get(values);
				rlComponent.setPosition(values);

				state.add(rlComponent);

			} else if (component instanceof Frame) {
				RlComponent rlComponent = new RlComponent();
				double[] values = new double[3];

				rlComponent.setName(component.getName());

				((Frame) component).getPosition().get(values);
				rlComponent.setPosition(values);

				values = new double[4];
				// Rotation is sent as a quaternion
				((Frame) component).getRotation().get(values);
				rlComponent.setOrientation(values);

				values = new double[6];
				((Frame) component).getVelocity().get(values);
				rlComponent.setTwist(values);

				state.add(rlComponent);
			} else
				throw new UnsupportedOperationException("not implemented");
		}
		return state;
	}

	ReentrantLock lock = new ReentrantLock();

	protected synchronized void waitForNextState() {
		while (!nextStateUpToDate) {
			try {
				wait();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	@Override
	public RlState setExcitations(ArrayList<Double> excitations) {
		Log.debug("setExcitations");
		if (getNextState) {
			// legacy lookahead path (state save/restore around manual advance)
			stageExcitations(excitations);
			nextStateUpToDate = false;
			Log.info("waiting for next state");
			waitForNextState();
			Log.info("next state done");
			return nextState;
		}
		RlState result = stepStrategy.applyExcitations(this, excitations);
		Log.debug("setExcitations done");
		return result;
	}

	/**
	 * Stage a new excitation vector for the next {@link #apply} call to
	 * push onto the muscles. Used by {@link StepStrategy} implementations.
	 */
	public void stageExcitations(ArrayList<Double> values) {
		this.excitationValues = values;
		this.excitersUpToDate = false;
	}

	/**
	 * Step size of the underlying mech system in seconds. Used by stepped
	 * strategies to compute how long to advance the simulation per action.
	 */
	public double getMechStepSize() {
		return myMech.getMaxStepSize();
	}

	/**
	 * Returns the {@link RootModel} that owns this controller. The reference
	 * is the same object as {@code myInverseModel} but typed for callers
	 * that need to invoke {@code advance(...)} or other RootModel methods
	 * directly.
	 */
	public artisynth.core.workspace.RootModel getRootModel() {
		return (artisynth.core.workspace.RootModel) myInverseModel;
	}

	/**
	 * Replace the current step strategy. Calls
	 * {@link StepStrategy#initialize(RlController)} on the new strategy.
	 * Intended to be called once at controller-setup time, not per step.
	 */
	public void setStepStrategy(StepStrategy strategy) {
		if (strategy == null) {
			throw new IllegalArgumentException("StepStrategy must not be null");
		}
		this.stepStrategy = strategy;
		strategy.initialize(this);
	}

	public StepStrategy getStepStrategy() {
		return stepStrategy;
	}

	public void setExcitationsZero() {
		for (int i = 0; i < exciters.size(); i++) {
			exciters.get(i).setExcitation(0.0);
		}
	}

	public void setRandomExcitations() {
		Random random = new Random();
		for (int i = 0; i < exciters.size(); ++i) {
			exciters.get(i).setExcitation(random.nextDouble());
		}

	}

	@Override
	public ArrayList<Double> getExcitations() {
		ArrayList<Double> exs = new ArrayList<Double>(exciters.size());

		for (ExcitationComponent m : this.exciters) {
			exs.add(m.getExcitation());
		}
		return exs;
	}
	
	@Override
	public ArrayList<Double> getMuscleForces() {
		ArrayList<Double> forces = new ArrayList<Double>(exciters.size());

		for (ExcitationComponent e : this.exciters) {
			MuscleExciter mex = (MuscleExciter)e;
			double force = 0;
			
			//TODO: make sure muscle not counted twice
			for (int i=0; i<mex.numTargets(); ++i) {
				ExcitationComponent ec = mex.getTarget(i);
				if (ec instanceof Muscle) {
					Muscle m = (Muscle)ec;				
					force += m.getForceNorm() - m.getPassiveForceNorm();
				} else if (ec instanceof MultiPointMuscle) {
					MultiPointMuscle m = (MultiPointMuscle)ec;				
					force += m.getForceNorm() - m.getPassiveForceNorm();
				}
					
			}			
			force /= mex.numTargets();
			forces.add(force);
		}
		return forces;
	}

	@Override
	public RlState resetState(boolean setExcitationsZero) {
		if (stepStrategy.isSynchronous()) {
			// In stepped mode the simulator is paused between actions, so it
			// is safe to mutate model state directly on the HTTP thread.
			myInverseModel.resetState();
			if (setExcitationsZero) {
				setExcitationsZero();
			}
			stepStrategy.onReset(this);
			Log.info("Reset complete (synchronous)");
			return getState();
		}

		// Free-run / legacy path: hand the reset over to the simulation
		// thread (it will pick it up in the next apply() call) and wait.
		resetZeroExcitations = setExcitationsZero;
		synchronized (resetLock) {
			resetPending = true;
			long deadline = System.currentTimeMillis() + 10_000;
			while (resetPending) {
				long remaining = deadline - System.currentTimeMillis();
				if (remaining <= 0) {
					Log.info("Reset timed out — simulation thread did not respond");
					resetPending = false;
					return getState();
				}
				try {
					resetLock.wait(remaining);
				} catch (InterruptedException e) {
					Thread.currentThread().interrupt();
					return getState();
				}
			}
		}
		Log.info("Reset complete");
		return resetResult != null ? resetResult : getState();
	}

	@Override
	public Map<String, Object> getInfo() {
		Map<String, Object> info = new LinkedHashMap<>();
		info.put("actionSize", getActionSize());
		info.put("obsSize", getObservationSize());
		info.put("stateSize", getStateSize());
		info.put("name", getName());
		return info;
	}
	
	@Override
	public String setSeed(int seed) {
		this.random.setSeed(seed);
		String message = "seed set to " + seed; 
		Log.debug(message);
		Log.info(message);
		return message;
	}
	
	@Override
	public String setTest(boolean isTest) {
		this.testTrial = isTest;		
		Log.info("Is Test: " + isTest);
		return "" + isTest;
	}
	
	@Override
	public boolean getTest() {
		return testTrial;
	}

	@Override
	public boolean isPlaying() {
		Main m = Main.getMain();
		return m != null && m.getScheduler().isPlaying();
	}

	@Override
	public void play() {
		Main m = Main.getMain();
		if (m != null) {
			Scheduler s = m.getScheduler();
			if (!s.isPlaying()) s.play();
		}
	}

	// --- video recording (GUI-only) -----------------------------------

	private String recordingName = null;

	@Override
	public String startRecording(String name, String outputDir, double fps) {
		Main main = Main.getMain();
		if (main == null || main.getViewer() == null) {
			throw new IllegalStateException(
				"Recording requires GUI mode — no viewer available.");
		}
		MovieMaker mm = main.getMovieMaker();
		if (outputDir != null && !outputDir.isEmpty()) {
			java.io.File dir = new java.io.File(outputDir);
			if (!dir.exists()) {
				dir.mkdirs();
			}
			mm.setMovieFolder(dir);
		}
		// Off-screen capture via the GL framebuffer object. Requires a
		// hardware-accelerated GL backend; software Mesa (e.g. WSL2 without
		// GPU passthrough) yields black frames because the FBO never gets
		// populated. See README → "Video recording requirements".
		int w = main.getViewer().getScreenWidth();
		int h = main.getViewer().getScreenHeight();
		mm.setCaptureArea(
			new java.awt.Rectangle(0, 0, w, h),
			new java.awt.Dimension(w, h));
		mm.setFrameRate(fps);
		mm.setMethod(MovieMaker.Method.FFMPEG);
		mm.resetFrameCounter();
		mm.setGrabbing(true);
		this.recordingName = (name != null && !name.isEmpty()) ? name : "recording";
		String msg = "Recording started → " + mm.getMovieFolderPath()
			+ " (" + w + "x" + h + ", fps=" + fps + ", method=FFMPEG)";
		Log.info(msg);
		return msg;
	}

	@Override
	public Map<String, Object> stopRecording() {
		Main main = Main.getMain();
		if (main == null || main.getViewer() == null) {
			throw new IllegalStateException(
				"Recording requires GUI mode — no viewer available.");
		}
		MovieMaker mm = main.getMovieMaker();
		mm.setGrabbing(false);
		Map<String, Object> info = new LinkedHashMap<>();
		// We deliberately skip mm.render(): it streams ffmpeg's stdout
		// through the MovieMakerDialog (which we never open) and would NPE.
		// The PNG frames are already on disk; the Python caller runs
		// ffmpeg itself and gets full control over codec / fps / output.
		String name = (this.recordingName != null) ? this.recordingName : "recording";
		try {
			int frames = mm.close();
			info.put("frames", frames);
			info.put("framePattern", "frame%05d.png");
			info.put("frameRate", mm.getFrameRate());
			info.put("outputDir", mm.getMovieFolderPath());
			info.put("baseName", name);
			Log.info("Recording stopped — " + info);
		}
		catch (Exception e) {
			Log.info("Movie close failed: " + e.getMessage());
			info.put("frames", -1);
			info.put("error", e.getMessage());
		}
		this.recordingName = null;
		return info;
	}

	/**
	 * Clears all terms and disposes storage
	 */
	public void dispose() {
		// System.out.println("tracking controller dispose()");
		targetPoints.clear();
		remove(targetPoints);
		targetFrames.clear();
		remove(targetFrames);
		sourcePoints.clear();
		remove(sourcePoints);
		sourceFrames.clear();
		remove(sourceFrames);

		for (@SuppressWarnings("unused")
		ExcitationComponent excCom : exciters) {

		}
		exciters.clear();
		remove(exciters);
	}

	protected boolean remove(ModelComponent comp) {
		return myComponents.remove(comp);
	}

	@Override
	public double getTime() {
		return myInverseModel.getTime();
	}
}