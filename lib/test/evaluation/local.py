from lib.test.evaluation.environment import EnvSettings
import os


def local_env_settings():
    settings = EnvSettings()

    here = os.path.dirname(os.path.abspath(__file__))
    repo_sub_dir = os.path.abspath(os.path.join(here, "..", "..", ".."))
    submission_dir = os.path.abspath(os.path.join(repo_sub_dir, ".."))

    settings.prj_dir = repo_sub_dir
    settings.save_dir = submission_dir

    settings.results_path = os.path.join(submission_dir, "test", "tracking_results")
    settings.segmentation_path = os.path.join(submission_dir, "test", "segmentation_results")
    settings.network_path = os.path.join(submission_dir, "test", "networks")
    settings.result_plot_path = os.path.join(submission_dir, "test", "result_plots")

    settings.otb_path = ""
    settings.nfs_path = ""
    settings.uav_path = ""
    settings.tpl_path = ""
    settings.vot_path = ""
    settings.got10k_path = ""
    settings.lasot_path = ""
    settings.trackingnet_path = ""
    settings.davis_dir = ""
    settings.youtubevos_dir = ""
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.tn_packed_results_path = ""

    return settings
