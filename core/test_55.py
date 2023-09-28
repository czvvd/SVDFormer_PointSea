import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.model_utils import PCViews
from models.SVDFormer import Model
from models.model_utils import fps_subsample

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
    test_metrics = AverageMeter(['CD','DCD','F1'])
    category_metrics = dict()
    mclass_metrics = AverageMeter(['CD','DCD','F1'])
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
        for batch_idx, (taxonomy_id, model_ids, data) in enumerate(t):
            taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
            with torch.no_grad():
                for k, v in data.items():
                    data[k] = utils.helpers.var_or_cuda(v)
                gt = data['gtcloud']
                _, npoints, _ = gt.size()

                # partial clouds from fixed viewpoints
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points = item)
                    partial = fps_subsample(partial, 2048)
                    partial_depth = torch.unsqueeze(render.get_img(partial), 1)
                    pcds_pred = model(partial.contiguous(),partial_depth)
                    cdl1,cdl2,f1 = calc_cd(pcds_pred[-1],gt,calc_f1=True)
                    dcd,_,_ = calc_dcd(pcds_pred[-1],gt)

                    test_metrics.update([cdl2.mean().item()*1e3,dcd.mean().item(),f1.mean().item()])
                    if taxonomy_id not in category_metrics:
                        category_metrics[taxonomy_id] = AverageMeter(['CD','DCD','F1'])
                    category_metrics[taxonomy_id].update([cdl2.mean().item()*1e3,dcd.mean().item(),f1.mean().item()])


                t.set_description('Test[%d/%d]  Average Metrics = %s' %
                             (batch_idx, n_samples,  ['%.4f' % l for l in test_metrics.avg()
                                                                                ]))

    # Record category results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy\t#Sample\t' + '\t'.join(test_metrics.items))

    for taxonomy_id in category_metrics:
        message = '{:s}\t{:d}\t'.format(taxonomy_id, category_metrics[taxonomy_id].count(0))
        message += '\t'.join(['%.4f' % value for value in category_metrics[taxonomy_id].avg()])
        mclass_metrics.update(category_metrics[taxonomy_id].avg())
        print(message)

    print('Overall\t{:d}\t'.format(test_metrics.count(0)) + '\t'.join(
        ['%.4f' % value for value in test_metrics.avg()]))
    print('MeanClass\t\t' + '\t'.join(['%.4f' % value for value in mclass_metrics.avg()]))


