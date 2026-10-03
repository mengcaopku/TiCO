import torch
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
from tico.constants import (
    DEFAULT_Ground_TOKEN,
    chat_template
)

def sample_bbox_topk(bboxes, scores, K=2, tau=0.3, device=None):
    """
    bboxes: list of bbox tuples, already sorted desc by score (your function does this)
    scores: list of floats, aligned with bboxes
    returns: chosen_bbox, chosen_idx (index in original bboxes list), logp_old (float)
    """
    M = len(bboxes)
    if M == 0:
        return None, None, None

    K = min(K, M)
    s = torch.tensor(scores[:K], dtype=torch.float, device=device).clamp_min(1e-12)

    logits = torch.log(s)
    probs = torch.softmax(logits / tau, dim=-1)

    dist = torch.distributions.Categorical(probs=probs)
    a = dist.sample()              
    logp = dist.log_prob(a)       

    chosen_idx = int(a.item())     
    return bboxes[chosen_idx], chosen_idx, float(logp.item())


def get_prediction_region_point(attn_scores, n_width, n_height,
                                activation_threshold=0.3,
                                return_all_regions=True,
                                rect_center=False):

    max_score = attn_scores[0].max().item()
    threshold = max_score * activation_threshold

    mask = attn_scores[0] > threshold
    valid_indices = torch.nonzero(mask).squeeze(-1)

    if valid_indices.numel() == 0:

        if return_all_regions:
            return None, [], [], []
        else:
            return None

    topk_values = attn_scores[0][valid_indices]
    topk_indices = valid_indices

    coords = []
    for idx in topk_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        coords.append((y, x, idx))

    coord_set = {(y, x): (idx, val.item()) for (y, x, idx), val in zip(coords, topk_values)}

    regions = []
    visited = set()
    for y, x, idx in coords:
        if idx in visited:
            continue
        region = []
        queue = [(y, x)]
        visited.add(idx)
        while queue:
            cy, cx = queue.pop(0)
            c_idx, c_val = coord_set[(cy, cx)]
            region.append((cy, cx, c_idx, c_val))
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in coord_set:
                    n_idx, n_val = coord_set[(ny, nx)]
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append((ny, nx))
        regions.append(region)

    region_scores, region_centers, region_points = [], [], []
    for region in regions:
        avg_score = sum(item[3] for item in region) / len(region)
        region_scores.append(avg_score)

        normalized_centers, weights = [], []
        y_coords, x_coords = set(), set()
        for y, x, _, score in region:
            cy = (y + 0.5) / n_height
            cx = (x + 0.5) / n_width
            normalized_centers.append((cx, cy))
            weights.append(score)
            y_coords.add(cy); x_coords.add(cx)

        region_points.append(normalized_centers)

        if not rect_center:
            total_w = sum(weights)
            wx = sum(pt[0] * w for pt, w in zip(normalized_centers, weights)) / total_w
            wy = sum(pt[1] * w for pt, w in zip(normalized_centers, weights)) / total_w
            region_centers.append((wx, wy))
        else:
            region_centers.append((sum(x_coords)/len(x_coords), sum(y_coords)/len(y_coords)))

    sorted_idx = sorted(range(len(region_scores)), key=lambda i: region_scores[i], reverse=True)
    sorted_scores = [region_scores[i] for i in sorted_idx]
    sorted_centers = [region_centers[i] for i in sorted_idx]
    sorted_points = [region_points[i] for i in sorted_idx]
    best_point = sorted_centers[0]

    if return_all_regions:
        return best_point, sorted_centers, sorted_scores, sorted_points
    else:
        return best_point


def get_activation_bboxes(attn_scores,
                          n_width: int,
                          n_height: int,
                          activation_threshold: float = 0.3,
                          normalized: bool = True,
                          connectivity: int = 4,
                          return_scores: bool = True):

    scores_flat = attn_scores[0]
    N = scores_flat.numel()
    assert N == n_width * n_height, "n_width*n_height must equal to the number of patches in attn_scores"

    max_score = scores_flat.max().item()
    threshold = max_score * activation_threshold
    mask = scores_flat > threshold
    valid_indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)

    if valid_indices.numel() == 0:
        return ([], []) if return_scores else []

    coords = []
    vals = scores_flat[valid_indices]
    for idx in valid_indices.tolist():
        y = idx // n_width
        x = idx % n_width
        coords.append((y, x, idx))
    coord_set = {(y, x): (idx, float(scores_flat[idx].item())) for (y, x, idx) in coords}

    if connectivity == 8:
        neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    else:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]

    visited = set()
    regions = [] 
    for y, x, idx in coords:
        if idx in visited:
            continue
        queue = [(y, x)]
        visited.add(idx)
        region = []
        while queue:
            cy, cx = queue.pop(0)
            c_idx, c_val = coord_set[(cy, cx)]
            region.append((cy, cx, c_idx, c_val))
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if (ny, nx) in coord_set:
                    n_idx, n_val = coord_set[(ny, nx)]
                    if n_idx not in visited:
                        visited.add(n_idx)
                        queue.append((ny, nx))
        regions.append(region)

    bboxes = []
    region_scores = []
    for region in regions:
        ys = [y for (y, x, _, _) in region]
        xs = [x for (y, x, _, _) in region]
        ymin, ymax = min(ys), max(ys)
        xmin, xmax = min(xs), max(xs)
        avg_score = sum(v for (_, _, _, v) in region) / len(region)

        if normalized:
            x0 = xmin / n_width
            y0 = ymin / n_height
            x1 = (xmax + 1) / n_width
            y1 = (ymax + 1) / n_height
            bboxes.append((x0, y0, x1, y1))
        else:
            bboxes.append((xmin, ymin, xmax + 1, ymax + 1))

        region_scores.append(avg_score)

    order = sorted(range(len(bboxes)), key=lambda i: region_scores[i], reverse=True)
    bboxes = [bboxes[i] for i in order]
    region_scores = [region_scores[i] for i in order]

    return (bboxes, region_scores) if return_scores else bboxes

def inference(conversation,
              model: Qwen2_5_VLForConditionalGeneration,
              tokenizer,
              data_processor,
              use_placeholder=False,
              topk=5,
              conv_iteration=1):

    pred = {
        "output_text": None,
        "n_width": None,
        "n_height": None,
        "attn_scores": None,
        "topk_points": None,
        "topk_values": None,
        "topk_points_all": None,
    }

    text = data_processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template
    )

    if use_placeholder:
        text += "<|im_start|>assistant"

    image_inputs, video_inputs = process_vision_info(conversation)
    inputs = data_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        results = model.generate(
            **inputs,
            max_new_tokens=2048 if not use_placeholder else 1,
            return_dict_in_generate=True,
            output_hidden_states=True,
            use_cache=True
        )

    input_ids = inputs["input_ids"][0]
    generated_ids = results.sequences[0][len(input_ids):]
    output_text = tokenizer.decode(
        generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    pred["output_text"] = output_text

    Ground_token_id = getattr(model.config, "Ground_token_id", None)
    if Ground_token_id is None:
        try:
            Ground_token_id = tokenizer.encode(DEFAULT_Ground_TOKEN)[0]
        except Exception:
            Ground_token_id = None

    if Ground_token_id is None:
        return pred

    if use_placeholder:
        Ground_mask = (inputs["input_ids"][0] == Ground_token_id)
    else:
        Ground_mask = (generated_ids[:-1] == Ground_token_id)
    if not Ground_mask.any():
        return pred

    if use_placeholder:
        decoder_hidden_states = results.hidden_states[0][-1][0]
    else:
        steps = [hs[-1][0] for hs in results.hidden_states[1:]]  
        decoder_hidden_states = torch.cat(steps, dim=0)   
        
    decoder_hidden_states = decoder_hidden_states[Ground_mask]

    image_pad_id = tokenizer.encode("<|image_pad|>")[0]
    image_mask = (inputs["input_ids"][0] == image_pad_id)
    image_embeds = results.hidden_states[0][0][0][image_mask] 

    attn_scores, _ = model.multi_patch_pointer_head(image_embeds, decoder_hidden_states)
    pred["attn_scores"] = attn_scores.tolist()

    _, n_h, n_w = (inputs["image_grid_thw"][0] // model.visual.spatial_merge_size).tolist()
    pred["n_width"] = n_w
    pred["n_height"] = n_h

    if conv_iteration == 1:
        best_point, sorted_centers, sorted_scores, sorted_points = get_prediction_region_point(
            attn_scores, n_w, n_h, return_all_regions=True, rect_center=False
        )
        bboxes, scores = get_activation_bboxes(attn_scores, n_w, n_h,
                                            activation_threshold=0.15,
                                            normalized=True, 
                                            connectivity=8,
                                            return_scores=True)

        if best_point is None:
            return pred

        topk_centers = sorted_centers[:topk]
        topk_scores = sorted_scores[:topk]
        topk_points_all = sorted_points[:topk]


        pred["topk_points"] = topk_centers
        pred["topk_values"] = topk_scores
        pred["topk_points_all"] = topk_points_all
        pred["bboxes"] = bboxes
        pred["bboxes_scores"] = scores

    return pred

def inference_rl_batch(
    prompt_inputs,
    temp_generation_config,
    model,
    tokenizer,
    use_placeholder=False,
    topk=5,
    conv_iteration=1,

    activation_threshold=0.15,
    connectivity=8,
    normalized=True,
    patch_sample_tau=1.0,       
):

    pred_list = []

    results = model.generate(
        **prompt_inputs,
        max_new_tokens=2048 if not use_placeholder else 1,
        generation_config=temp_generation_config,
        temperature=1.0,
        use_cache=True,
        return_dict_in_generate=True,
        output_hidden_states=True,
    )

    model.multi_patch_pointer_head.eval()
    sequences = results.sequences
    hidden_states = results.hidden_states

    if isinstance(prompt_inputs, list):
        per_prefix_lens = []
        for pi in prompt_inputs:
            ids = pi["input_ids"]
            if isinstance(ids, torch.Tensor):
                per_prefix_lens.append(ids.shape[-1] if ids.dim() > 1 else ids.numel())
            else:
                per_prefix_lens.append(len(ids))
    else:
        first_ids = prompt_inputs["input_ids"][0]
        per_prefix_lens = [len(first_ids) for _ in range(sequences.shape[0])]

    generated_ids_list = []
    for out_idx, seq_tensor in enumerate(sequences):
        prefix_len = per_prefix_lens[out_idx] if out_idx < len(per_prefix_lens) else per_prefix_lens[0]
        gen_tail = seq_tensor[prefix_len:]
        generated_ids_list.append(gen_tail.cpu().tolist())

    output_text_stage1_list = tokenizer.batch_decode(
        generated_ids_list, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )

    Ground_token_id = getattr(model.config, "Ground_token_id", None)
    if Ground_token_id is None:
        try:
            Ground_token_id = tokenizer.encode("<|Ground|>")[0]
        except Exception:
            Ground_token_id = None

    image_pad_id = tokenizer.encode("<|image_pad|>")[0]

    def _patch_idx_to_yx(idx: int, n_w: int):
        return idx // n_w, idx % n_w

    batch_outputs = sequences.shape[0]
    for out_idx in range(batch_outputs):
        pred = {}
        pred["output_text"] = output_text_stage1_list[out_idx]
        pred["generated_ids_list"] = sequences[out_idx]
        pred["valid_pointer"] = False

        prefix_len = per_prefix_lens[out_idx] if out_idx < len(per_prefix_lens) else per_prefix_lens[0]

        this_input_ids = prompt_inputs["input_ids"][0]

        gen_tail = torch.tensor(generated_ids_list[out_idx], device=sequences.device)

        if Ground_token_id is None or gen_tail.numel() <= 1:
            pred_list.append(pred)
            continue

        Ground_mask = (gen_tail[:-1] == Ground_token_id)
        if not Ground_mask.any():
            pred_list.append(pred)
            continue

        steps = [hs[-1][0] for hs in hidden_states[1:]]        
        decoder_hidden_all = torch.cat(steps, dim=0)            
        decoder_hidden_states = decoder_hidden_all[Ground_mask] 
        if decoder_hidden_states.numel() == 0:
            pred_list.append(pred)
            continue

        ptr_positions = torch.nonzero(Ground_mask, as_tuple=False).squeeze(-1)
        ptr_pos = int(ptr_positions[0].item())  
        pred["pointer_pos_in_gen"] = ptr_pos
        pred["prefix_len"] = int(prefix_len)
        pred["pointer_pos_in_full"] = int(prefix_len + ptr_pos)

        image_mask = (this_input_ids == image_pad_id)
        if not bool(image_mask.any().item()):
            pred_list.append(pred)
            continue
        image_embeds = hidden_states[0][0][0][image_mask]

        attn_scores, _ = model.multi_patch_pointer_head(image_embeds, decoder_hidden_states)
        pred["attn_scores"] = attn_scores.tolist()

        try:
            grid = prompt_inputs["image_grid_thw"][0]
            _, n_h, n_w = (grid // model.visual.spatial_merge_size).tolist()
            pred["n_width"] = n_w
            pred["n_height"] = n_h
        except Exception:
            pred_list.append(pred)
            continue

        if conv_iteration == 1:
            best_point, sorted_centers, sorted_scores, sorted_points = get_prediction_region_point(
                attn_scores, n_w, n_h, return_all_regions=True, rect_center=False
            )
            bboxes, scores = get_activation_bboxes(
                attn_scores,
                n_w,
                n_h,
                activation_threshold=activation_threshold,
                normalized=normalized,
                connectivity=connectivity,
                return_scores=True,
            )

            if best_point is not None:
                pred["topk_points"] = sorted_centers[:topk]
                pred["topk_values"] = sorted_scores[:topk]
                pred["topk_points_all"] = sorted_points[:topk]
                bboxes = bboxes[:topk]
                scores = scores[:topk]
                pred["bboxes"] = bboxes
                pred["bboxes_scores"] = scores

            p_patch = attn_scores[0].squeeze() 
            p_patch = p_patch.clamp(min=0.0)
            p_patch = p_patch / p_patch.sum().clamp_min(1e-12)

            if patch_sample_tau is not None and patch_sample_tau > 0 and abs(patch_sample_tau - 1.0) > 1e-6:
                logits = torch.log(p_patch.clamp_min(1e-12))
                p_patch = torch.softmax(logits / float(patch_sample_tau), dim=-1)

            dist = torch.distributions.Categorical(probs=p_patch)
            a_patch = dist.sample()                 
            logp_old = dist.log_prob(a_patch)      
            a_patch_idx = int(a_patch.item())

            pred["ptr_action_patch"] = a_patch_idx
            pred["ptr_logp_patch_old"] = float(logp_old.item())

            chosen_bbox = None
            chosen_bbox_idx = -1

            if bboxes is not None and len(bboxes) > 0:
                py, px = _patch_idx_to_yx(a_patch_idx, n_w)

                for bi, bbox in enumerate(bboxes):
                    x0, y0, x1, y1 = bbox
                    xmin = int(x0 * n_w)
                    ymin = int(y0 * n_h)
                    xmax = int(x1 * n_w) - 1
                    ymax = int(y1 * n_h) - 1
                    if xmin <= px <= xmax and ymin <= py <= ymax:
                        chosen_bbox = bbox
                        chosen_bbox_idx = bi
                        break

                if chosen_bbox is None:
                    chosen_bbox = bboxes[0]
                    chosen_bbox_idx = 0

            pred["chosen_bbox"] = chosen_bbox
            pred["chosen_bbox_idx"] = chosen_bbox_idx
            pred["valid_pointer"] = bool(chosen_bbox is not None)

        else:
            pred["valid_pointer"] = True

        pred_list.append(pred)

    return pred_list