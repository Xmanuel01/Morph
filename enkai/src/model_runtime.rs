use crate::model::{
    model_list, model_load, model_loaded, model_manifest_args, model_promote_like,
    model_pull_remote, model_push_remote, model_register, model_retire, model_sync_remote_state,
    model_unload, model_verify_signature, ModelCommandManifest, ModelManifestCommand,
};

pub(crate) fn execute_model_command_manifest(manifest: &ModelCommandManifest) -> i32 {
    let args = model_manifest_args(manifest);
    match &manifest.command {
        ModelManifestCommand::Register { .. } => model_register(&args),
        ModelManifestCommand::List { .. } => model_list(&args),
        ModelManifestCommand::Load { .. } => model_load(&args),
        ModelManifestCommand::Unload { .. } => model_unload(&args),
        ModelManifestCommand::Loaded { .. } => model_loaded(&args),
        ModelManifestCommand::Push { .. } => model_push_remote(&args),
        ModelManifestCommand::Pull { .. } => model_pull_remote(&args),
        ModelManifestCommand::VerifySignature { .. } => model_verify_signature(&args),
        ModelManifestCommand::PromoteRemote { .. } => model_sync_remote_state("promote", &args),
        ModelManifestCommand::RetireRemote { .. } => model_sync_remote_state("retire", &args),
        ModelManifestCommand::RollbackRemote { .. } => model_sync_remote_state("rollback", &args),
        ModelManifestCommand::Promote { .. } => model_promote_like("promote", &args),
        ModelManifestCommand::Retire { .. } => model_retire(&args),
        ModelManifestCommand::Rollback { .. } => model_promote_like("rollback", &args),
    }
}
