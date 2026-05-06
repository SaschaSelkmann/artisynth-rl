package artisynth.core.rl;

import spark.Request;
import spark.Response;
import spark.Route;
import static artisynth.core.rl.JsonUtil.*;
import static spark.Spark.*;

import com.google.gson.Gson;

public class RlRestApi {
	RlControllerInterface rlController;
	int serverPort;

	public RlRestApi(RlControllerInterface rlController, int serverPort) {
		this.rlController = rlController;
		this.serverPort = serverPort;
		port(serverPort);

		get("/", (req, res) -> "ArtiSynth RL REST API v2");

		// --- info: returns all space sizes in one round-trip ---
		get("/info", (req, res) -> rlController.getInfo(), json());

		// --- state ---
		get("/state", (req, res) -> rlController.getState(), json());
		get("/time",  (req, res) -> rlController.getTime(), json());

		// --- space sizes (kept for backward compat) ---
		get("/obsSize",    (req, res) -> rlController.getObservationSize(), json());
		get("/stateSize",  (req, res) -> rlController.getStateSize(), json());
		get("/actionSize", (req, res) -> rlController.getActionSize(), json());

		// --- excitations ---
		get("/excitations",  (req, res) -> rlController.getExcitations(), json());
		post("/excitations", setExcitations, json());

		// --- control ---
		post("/reset",   (req, res) -> rlController.resetState(Boolean.parseBoolean(req.body())), json());
		post("/setSeed", (req, res) -> rlController.setSeed(Integer.parseInt(req.body())), json());
		post("/setTest", (req, res) -> rlController.setTest(Boolean.parseBoolean(req.body())), json());

		// --- scheduler ---
		get("/isPlaying", (req, res) -> rlController.isPlaying(), json());
		post("/play", (req, res) -> { rlController.play(); return true; }, json());

		after((req, res) -> res.type("application/json"));

		exception(IllegalArgumentException.class, (e, req, res) -> {
			res.status(400);
			res.body(toJson(new ResponseError(e)));
		});
	}

	public Route setExcitations = (Request request, Response response) -> {
		Log.debug("setExcitations length:" + request.contentLength());
		Gson gson = new Gson();
		RlMuscleProps rlExcitations = gson.fromJson(request.body(), RlMuscleProps.class);
		return this.rlController.setExcitations(rlExcitations.getProps());
	};
}
