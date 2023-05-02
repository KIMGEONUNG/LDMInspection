import torch

# path1 = "logs/2023-04-26T17-17-30_T001-B/checkpoints/trainstep_checkpoints/epoch=000000-step=000009999.ckpt"
path1 = "logs/2023-04-27T01-19-14_T001-B/checkpoints/trainstep_checkpoints/epoch=000000-step=000000099.ckpt"
path2 = "models/first_stage_models/kl-f8/model.ckpt"

original = torch.load(path2, map_location=torch.device('cpu'))["state_dict"]
trained = torch.load(path1, map_location=torch.device('cpu'))["state_dict"]

# for key in original:
#     if "encoder." in key:
#         a = trained[key]
#         b = trained[key.replace("encoder.", "encoder_fix.")]
#         equal = torch.equal(a, b)
#         # if not equal:
#         print(key, ":", equal)
            
for key in original:
    if "encoder." in key:
        a = original[key]
        b = trained[key.replace("encoder.", "encoder_fix.")]
        equal = torch.equal(a, b)
        print(key, ":", equal)
    if "quant_conv." in key and "post" not in key:
        a = original[key]
        b = trained[key.replace("quant_conv.", "quant_conv_fix.")]
        equal = torch.equal(a, b)
        print(key, ":", equal)
