class PipelineValidator:
    def __init__(self, pipeline_spec: Dict, debug_mode: bool = False):
        """
        Initialize validator with pipeline specification and optional debug mode
        """
        self.pipeline_spec = pipeline_spec
        self.current_stage = 0
        self.debug_mode = debug_mode

        # Execution statistics
        self.execution_stats = {
            "steps_completed": 0,
            "validation_attempts": 0,
            "validation_failures": 0,
            "start_time": None,
            "end_time": None,
            "step_times": {},
            "validation_details": {}  # Store detailed validation results
        }

        # Track state for each stage
        self.expected_results = {}  # Store expected dataframes for each stage
        self.agent_results = {}     # Store agent-produced dataframes for each stage
        self.stage_hashes = {}      # Store hashes of expected dataframes
        self.stage_fingerprints = {}  # Store structure fingerprints

    def start(self):
        """Start pipeline execution tracking"""
        self.execution_stats["start_time"] = time.time()

    def set_expected_result(self, stage_id: str, df: pd.DataFrame) -> None:
        """
        Set the expected result for a stage
        """
        # Store hash of the dataframe
        df_hash = self._compute_dataframe_hash(df)
        self.stage_hashes[stage_id] = df_hash

        # Store structure fingerprint
        self.stage_fingerprints[stage_id] = self._compute_structure_fingerprint(df)

        # Store full dataframe if in debug mode
        if self.debug_mode:
            self.expected_results[stage_id] = df.copy()

    def validate_step(self, stage_id: str, agent_df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate that the agent's dataframe matches expected result for this stage
        """
        self.execution_stats["validation_attempts"] += 1

        # Check if this is the correct next stage
        expected_stage_id = f"stage_{self.current_stage}"
        if stage_id != expected_stage_id:
            result = {
                "success": False,
                "message": f"Invalid stage sequence. Expected {expected_stage_id}, got {stage_id}",
                "details": {"sequence_error": True}
            }
            self.execution_stats["validation_failures"] += 1
            self.execution_stats["validation_details"][stage_id] = result
            return False, result

        # If we don't have expected results for this stage yet, set them from the agent's result
        if stage_id not in self.stage_hashes:
            self.set_expected_result(stage_id, agent_df)
            result = {
                "success": True,
                "message": "Baseline set for future validation",
                "details": {"baseline_set": True}
            }
        else:
            # Compute validation results
            validation_results = self._validate_dataframe(stage_id, agent_df)
            success = validation_results["success"]

            result = {
                "success": success,
                "message": "Validation successful" if success else "Validation failed",
                "details": validation_results
            }

            if not success:
                self.execution_stats["validation_failures"] += 1

        # Store agent's dataframe if in debug mode
        if self.debug_mode:
            self.agent_results[stage_id] = agent_df.copy()

        # Update execution stats
        if result["success"]:
            self.execution_stats["steps_completed"] += 1
            self.execution_stats["step_times"][stage_id] = time.time()
            self.current_stage += 1

        self.execution_stats["validation_details"][stage_id] = result
        return result["success"], result

    def _validate_dataframe(self, stage_id: str, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive validation of a dataframe against expected results
        """
        results = {
            "success": True,
            "hash_match": False,
            p.allclose(s1.values, s2.values, rtol=1e-5, atol=1e-8, equal_nan=True):
                all_match = False
                # Find different rows
                diff_idxs = ~np.isclose(s1.values, s2.values, rtol=1e-5, atol=1e-8, equal_nan=True)
                diff_count = np.sum(diff_idxs)

                # Include sample of differences
                if diff_count > 0:
                    sample_idx = np.where(diff_idxs)[0][0]
                    differences[col] = {
                        "diff_count": int(diff_count),
                        "diff_percent": float(diff_count / len(s1) * 100),
                        "example_idx": int(sample_idx),
                        "expected": float(s1.iloc[sample_idx]),
                        "actual": float(s2.iloc[sample_idx])
                    }

        # Compare non-numeric columns
        non_numeric_cols = [col for col in common_cols if col not in numeric_cols]

        for col in non_numeric_cols:
            # Fill NaNs with a placeholder for comparison
            s1 = df1[col].fillna("")
            s2 = df2[col].fillna("")

            if not s1.equals(s2):
                all_match = False
                # Find different rows
                diff_idxs = (s1 != s2)
                diff_count = diff_idxs.sum()

                # Include sample of differences
                if diff_count > 0:
                    sample_idx = diff_idxs.idxmax()
                    differences[col] = {
                        "diff_count": int(diff_count),
                        "diff_percent": float(diff_count / len(s1) * 100),
                        "example_idx": int(sample_idx),
                        "expected": str(s1.iloc[sample_idx]),
                        "actual": str(s2.iloc[sample_idx])
                    }

        return all_match, differences

    def complete(self) -> Dict:
        """Complete pipeline execution and return stats"""
        self.execution_stats["end_time"] = time.time()
        self.execution_stats["total_time"] = self.execution_stats["end_time"] - self.execution_stats["start_time"]
        self.execution_stats["completed"] = self.current_stage >= len(self.pipeline_spec["stages"])
        return self.execution_stats

    def get_validation_report(self) -> Dict:
        """
        Generate a comprehensive validation report
        """
        report = {
            "summary": {
                "total_stages": len(self.pipeline_spec["stages"]),
                "completed_stages": self.execution_stats["steps_completed"],
                "validation_attempts": self.execution_stats["validation_attempts"],
                "validation_failures": self.execution_stats["validation_failures"],
                "success_rate": (
                    (self.execution_stats["validation_attempts"] - self.execution_stats["validation_failures"]) /
                    self.execution_stats["validation_attempts"] if self.execution_stats["validation_attempts"] > 0 else 0
                ) * 100
            },
            "timing": {
                "total_time": round(self.execution_stats.get("total_time", 0), 2),
                "stage_times": {k: round(v - self.execution_stats["start_time"], 2)
                               for k, v in self.execution_stats["step_times"].items()}
            },
            "stage_results": self.execution_stats["validation_details"]
        }

        # Add overall result
        report["success"] = report["summary"]["validation_failures"] == 0

        return report