import numpy as np
"""params={}
params['allowed_attrs_file']='./Flipkart/Attribute_allowedvalues.npy'
params['category_attrs_mapping']='./Flipkart/vertical_attributes.npy'
"""
def decode_outputs(threshold,values_obj,category_mapping,cat_output,val_output):
    """ Remember not to do shuffle and keep the same batch size for both
        Takes as input :
        params: for eval_threshold key
        values_obj: The loaded class Values obj
        category_mapping: The dictionary mapping categories encodings to the strings.
        cat_output: Tensor of size(batch_size,26)
        val_output: Tensor of size(batch_size,802)
        Returns the output after decoding:
        To be passed directly to get_outputs function
    """


    category_list=cat_output.argmax(dim=1)
    category_list=[category_mapping[i.item()] for i in category_list]
    
    m,_=val_output.shape
    values_list=[]
    for batch in range(m):
        ans=list((torch.where(val_output[batch]>threshold)[0]))
        ans=[values_obj.index2word[i.item()] for i in ans]
        values_list.append(ans)
    
    return category_list,values_list

def get_allowed_values(file_name):
    """ Takes as input the file mapping attribute to values returns the dict of all allowed values based on attrs"""
    attibute_allowedvalue_dict = np.load(file_name,allow_pickle=True).tolist()
    return attibute_allowedvalue_dict

def get_allowed_cat_attrs(file_name):
    """ Takes as input the file mapping category to attribute  returns the dict of all allowed attributes based on vat """
    vertical_value_dict = np.load(file_name,allow_pickle=True).tolist()
    return vertical_value_dict

def get_outputs(category_attr_mapping,attr_values_mapping,pred_categories, pred_values):
    """ Takes as input the params dict, decoded form of category  and decode form of attribute values ( As a list of items)
        Returns the prediction list as a dict of attrs
        Usage:
        pred_categories=['capri']
        pred_values=[['beach',
          'casual',
          'ethnic','above knee',
          'ankle','sleepwear',
         'sportswear']]

    """
    n_items=len(pred_categories)
    #category_attr_mapping=get_allowed_cat_attrs(params['category_attrs_mapping'])
    #attr_values_mapping=get_allowed_values(params['allowed_attrs_file'])
    predictions=[]
    for key in category_attr_mapping.keys():
        category_attr_mapping[key]=[i for i in category_attr_mapping[key] if i!='vertical']
    for key in attr_values_mapping.keys():
        attr_values_mapping[key]=set(attr_values_mapping[key])
    for i in range(n_items):
        tempdict={}
        preds=set(pred_values[i])
        for attr in category_attr_mapping[pred_categories[i]]:
            allowed_vals_cats=attr_values_mapping[attr]
            tempdict[attr]=list(allowed_vals_cats.intersection(preds))
        predictions.append(tempdict)
    return predictions



def get_iou(*dicts):
    """Takes input as ground truth dict and predicted attributes dict """
    comm_keys = dicts[0].keys()
    for d in dicts[1:]:
        # intersect keys first
        comm_keys &= d.keys()
    # then build a result dict with nested comprehension
    score=[]
    for key in comm_keys:
        gt=set(dicts[0][key])
        pred=set(dicts[1][key])
        union=gt.union(pred)
        inter=gt.intersection(pred)
        score.append(len(inter)/len(union))
#     To average the score dividing by the ground truth length
    return sum(score)/len(dicts[0])

# For calculating the score
def evaluate_model_scores(ground_truth,predictions):
    """
    The function takes as input two list of dictionaries where each dictionary 
    contains keys as attributes and values as the attribute values
    Returns average scores
    Usage:
    dict1={'a':[2,3],'b':[6,7] }
    dict2={'c':[7],'a':[2,3],'b':[9] }
    gt=[]
    pred=[]
    gt.append(dict1)
    pred.append(dict2)
    gt.append(dict2)
    pred.append(dict1)

    evaluate_model_scores(gt,pred)
    Returns the average score
    """
    n_items=len(ground_truth)
    total_scores=[]
    for i in range(n_items):
        total_scores.append( get_iou(ground_truth[i],predictions[i]) )
    print(total_scores)
    return sum(total_scores)/len(total_scores)

