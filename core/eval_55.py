import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.model_utils import PCViews
from models.SVDFormer import Model

def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
        utils.data_loaders.DatasetSubset.TEST),
                                                  batch_size=1,
                                                  num_workers=4,
                                                  collate_fn=utils.data_loaders.collate_fn_55,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Setup networks and initialize networks
    if model is None:
        model = Model(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD', 'DCD', 'F1'])
    test_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    category_metrics = dict()
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    # Eval settings
    crop_ratio = {
        'easy': 1 / 4,
        'median': 1 / 2,
        'hard': 3 / 4
    }
    choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
              torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
              torch.Tensor([-1, -1, -1])]

    mode = cfg.CONST.mode

    print('Start evaluating (mode: {:s}) ...'.format(mode))

    # Testing loop
    with tqdm(test_data_loader) as t:
        for batch_idx, (taxonomy_ids, model_ids, data) in enumerate(t):
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                # partial = data['partial_cloud']
                gt = data['gtcloud']
                _, npoints, _ = gt.size()
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    partial = fps_subsample(partial, 2048)
                    partial_depth = torch.unsqueeze(render.get_img(partial), 1)
                    pcds_pred = model(partial.contiguous(),partial_depth)
                    cdl1, cdl2, f1 = calc_cd(pcds_pred[-1], gt, calc_f1=True)
                    dcd, _, _ = calc_dcd(pcds_pred[-1], gt)

                    cd = cdl2.mean().item() * 1e3
                    dcd = dcd.mean().item()
                    f1 = f1.mean().item()

                    _metrics = [cd, dcd, f1]
                    test_losses.update([cd, dcd, f1])

                    test_metrics.update(_metrics)

                    t.set_description('Test[%d/%d]  Losses = %s Metrics = %s' %(batch_idx, n_samples,  ['%.4f' % l for l in test_losses.avg()
                                                                                ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)
