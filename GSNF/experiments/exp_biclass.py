from experiments import BaseExperiment


class Exp_BiClass(BaseExperiment):
    def validation_step(self, epoch):
        results = self.compute_results_all_batches(self.dlval)
        self.logger.info(f"val_auroc={results['auroc']:.5f}")
        self.logger.info(f"val_auprc={results['auprc']:.5f}")
        self.logger.info(f"val_ce_loss={results['ce_loss']:.5f}")
        self.logger.info(f"val_forward_time={results['forward_time']:.5f}")
        results["monitor_loss"] = results["val_loss"]
        return results

    def test_step(self):
        results = self.compute_results_all_batches(self.dltest)
        self.logger.info(f"test_auroc={results['auroc']:.5f}")
        self.logger.info(f"test_auprc={results['auprc']:.5f}")
        self.logger.info(f"test_ce_loss={results['ce_loss']:.5f}")
        self.logger.info(f"test_forward_time={results['forward_time']:.5f}")
        results["monitor_loss"] = results["val_loss"]
        return results
