#ifdef __cplusplus
extern "C" {
#endif
#include "EXTERN.h"
#include "perl.h"
#include "XSUB.h"
#ifdef __cplusplus
}
#endif

#include "ppport.h"
#include <string>
#include <vector>
#include <map>
#include <cfloat>
#include <cmath>

typedef std::map<int, int> IntToIntMap;
typedef std::map<int, double> IntToDoubleMap;
typedef std::map<int, std::map<int, double> > Int2ToDoubleMap;
typedef std::map<std::string, int> StrToIntMap;
typedef std::map<std::string, double> StrToDoubleMap;

typedef std::vector<double> DoubleVector;
typedef std::vector<std::vector<double> > Double2DVector;

typedef std::vector<IntToDoubleMap> IDMapVector;

class NBClustering{
  public:
    NBClustering();
    ~NBClustering();
    void AddInstance(const StrToDoubleMap &doc);
    void AddLabeledInstance(const StrToDoubleMap &doc, const std::string &label);
    void Train(const int cluster_num, int max_iteration, double epsilon, double alpha, int seed);
    StrToDoubleMap Predict(const StrToDoubleMap &doc);
  private:
    void initializePrior(DoubleVector *prior, int num, double alpha);
    void initializeParam(double *array, int num);
    void normalizeParam(double *array, int num, DoubleVector *prior);
    double calcLogPrior(double *cls_param, double **wrd_param, int cnum, int wnum);
    int numDoc;
    StrToIntMap cdict;
    StrToIntMap wdict;
    IDMapVector data;
    IntToIntMap ccount;
    Int2ToDoubleMap wcount;
    DoubleVector cparam;
    Double2DVector wparam;
    DoubleVector cprior;
    Double2DVector wprior;
};

NBClustering::NBClustering()
  : cdict(), wdict(), data(), ccount(), wcount(), cparam(), wparam(), cprior(), wprior()
{
}

NBClustering::~NBClustering()
{
}

void
NBClustering::AddInstance(const StrToDoubleMap &instance)
{
  IntToDoubleMap datum;

  for (StrToDoubleMap::const_iterator it = instance.begin(); it != instance.end(); ++it) {
    if (wdict.find(it->first) == wdict.end()) {
      int wnum = wdict.size();
      wdict[it->first] = wnum;
    }
    int w = wdict[it->first];
    datum[w] = it->second;
  }

  data.push_back(datum);

  return;
}

void
NBClustering::AddLabeledInstance(const StrToDoubleMap &instance, const std::string &label)
{

  if (cdict.find(label) == cdict.end()) {
    int cnum = cdict.size();
    cdict[label] = cnum;
  }

  int c = cdict[label];
  ccount[c] += 1;

  for (StrToDoubleMap::const_iterator it = instance.begin(); it != instance.end(); ++it) {
    if (wdict.find(it->first) == wdict.end()) {
      int wnum = wdict.size();
      wdict[it->first] = wnum;
    }
    int w = wdict[it->first];
    wcount[c][w] += it->second;
  }

  return;
}


void
NBClustering::Train(const int cluster_num, int max_iteration, double epsilon, double alpha, int seed)
{
  /*
  int max_iteration = 100;
  double epsilon = 1e-10;
  double alpha = 1.0;
  int seed = 1000;
  */

  srand(seed);

  int cnum = (cdict.size() < 2) ? cluster_num : cdict.size(); 
  int wnum = wdict.size();

  double *cls_param  = new double[cnum];
  double **wrd_param = new double*[cnum];
  double *cls_exp  = new double[cnum];
  double **wrd_exp = new double*[cnum];
  double *cls_tmp  = NULL;
  double **wrd_tmp = NULL;

  initializeParam(cls_param, cnum);
  std::fill_n(cls_exp, cnum, 0.0);

  for (int c = 0; c < cnum; ++c) {
    wrd_param[c] = new double[wnum];
    wrd_exp[c]   = new double[wnum];
    initializeParam(wrd_param[c], wnum);
    std::fill_n(wrd_exp[c], wnum, 0.0);
  }

  if (cdict.size() < 2) {
    initializePrior(&cprior, cnum, alpha);

    char buf[1024];
    for (int c = 0; c < cluster_num; ++c) {
      sprintf(buf, "%d", c);
      cdict[std::string(buf)] = c;

      std::vector<double> tmp;
      initializePrior(&tmp, wnum, alpha);
      wprior.push_back(tmp);
    }
  } else {
    cprior.assign(cnum, alpha);
    
    for (IntToIntMap::iterator cit = ccount.begin(); cit != ccount.end(); ++cit) {
      int c = cit->first;
      cprior[c] += cit->second;

      std::vector<double> tmp;
      tmp.assign(wnum, alpha);

      for (IntToDoubleMap::iterator wit = wcount[c].begin(); wit != wcount[c].end(); ++wit) {
        int w = wit->first;
        tmp[w] += wit->second;
      }
      wprior.push_back(tmp);
    }
  }

  bool converged = false;
  int iter = 0;
  double prev_logprobs = -DBL_MAX;
  double curr_logprobs = -DBL_MAX;
  std::vector<double> cond_prob(cnum);

  while (!converged && iter <= max_iteration) {
    ++iter;

    double logprobs = 0.0;

    // Expectation step
    for (int d = 0; d < data.size(); ++d) {
      IntToDoubleMap datum = data[d];

      double scaled_sum = 0.0;
      double max_val = -DBL_MAX;

      for (int c = 0; c < cnum; ++c) {
        cond_prob[c] = log(cls_param[c]);

        for (IntToDoubleMap::iterator it = datum.begin(); it != datum.end(); ++it) {
          cond_prob[c] += it->second * log(wrd_param[c][it->first]);
        }
        if (cond_prob[c] > max_val) {
          max_val = cond_prob[c];
        }
      }

      for (int c = 0; c < cnum; ++c) {
        cond_prob[c] = exp(cond_prob[c] - max_val);
        scaled_sum += cond_prob[c];
      }

      logprobs += max_val + log(scaled_sum);

      for (int c = 0; c < cnum; ++c) {
        cond_prob[c] /= scaled_sum;
        cls_exp[c] += cond_prob[c];
        for (IntToDoubleMap::iterator it = datum.begin(); it != datum.end(); ++it) {
          wrd_exp[c][it->first] += it->second * cond_prob[c];
        }
      }
    }

    double logprior = calcLogPrior(cls_param, wrd_param, cnum, wnum);
    logprobs += logprior;

    // Maximization step
    normalizeParam(cls_exp, cnum, &cprior);
    for (int c = 0; c < cnum; ++c) {
      normalizeParam(wrd_exp[c], wnum, &(wprior[c]));
    }

    if (fabs(logprobs - curr_logprobs) < fabs(curr_logprobs) * epsilon) {
      converged = true;
    } else if (logprobs < curr_logprobs) {
      fprintf(stderr, "decrese: %15.10f to %15.10f\n", curr_logprobs, logprobs);
      converged = true;
    } else if (data.size() == 0) {
      converged = true;
    }

    fprintf(stderr, "%5d: %15.10f\n", iter, logprobs);

    prev_logprobs = curr_logprobs;
    curr_logprobs = logprobs;

    cls_tmp = cls_param; cls_param = cls_exp; cls_exp = cls_tmp;
    wrd_tmp = wrd_param; wrd_param = wrd_exp; wrd_exp = wrd_tmp;
    for (int c = 0; c < cnum; ++c) {
      std::fill_n(wrd_exp[c], wnum, 0.0);
    }
    std::fill_n(cls_exp, cnum, 0.0);
  }

  cparam.clear();
  wparam.clear();

  for (int c = 0; c < cnum; ++c) {
    cparam.push_back(cls_param[c]);

    std::vector<double> tmp_vec;
    for (int w = 0; w < wnum; ++w) {
      tmp_vec.push_back(wrd_param[c][w]);
    }
    wparam.push_back(tmp_vec);

    delete[] wrd_param[c]; wrd_param[c] = NULL;
    delete[] wrd_exp[c];   wrd_exp[c]   = NULL;
  }
  delete[] wrd_param; wrd_param = NULL;
  delete[] wrd_exp;   wrd_exp   = NULL;

  return;
}

void
NBClustering::initializePrior(DoubleVector *prior, int num, double alpha)
{
  (*prior).clear();
  (*prior).assign(num, alpha);
}

void
NBClustering::initializeParam(double *array, int num)
{
  double sum = 0.0;
  for (int i = 0; i < num; ++i) {
    double tmp = static_cast<double>(rand()) / RAND_MAX; 
    array[i] = 1.0 + tmp;
    sum += array[i];
  }
  for (int i = 0; i < num; ++i) {
    array[i] /= sum;
  }
  return;
}

void
NBClustering::normalizeParam(double *array, int num, DoubleVector *prior)
{
  double sum = 0.0;
  for (int i = 0; i < num; ++i) {
    array[i] += (*prior)[i];
    sum += array[i];
  }
  for (int i = 0; i < num; ++i) {
    array[i] /= sum;
  }
  return;
}

double
NBClustering::calcLogPrior(double *cls_param, double **wrd_param, int cnum, int wnum)
{
  double logprior = 0.0;
  for (int c = 0; c < cnum; ++c) {
    if (cprior[c] > 0) {
      logprior += cprior[c] * log(cls_param[c]);
    }
    for (int w = 0; w < wnum; ++w) {
      if (wprior[c][w]) {
        logprior += wprior[c][w] * log(wrd_param[c][w]);
      }
    }
  }
  return logprior;
}


StrToDoubleMap
NBClustering::Predict(const StrToDoubleMap &doc)
{
  StrToDoubleMap result;
  double max_score = -DBL_MAX;

  for (StrToIntMap::iterator cit = cdict.begin(); cit != cdict.end(); ++cit) {
    std::string cluster = cit->first;
    int c = cdict[cluster];
    result[cluster] = log(cparam[c]);

    for (StrToDoubleMap::const_iterator wit = doc.begin(); wit != doc.end(); ++wit) {
      std::string word = wit->first;
      double freq = wit->second;

      if (wdict.find(word) == wdict.end()) {
        continue;
      }
      int w = wdict[word];

      result[cluster] += freq * log(wparam[c][w]);
    }

    if (result[cluster] > max_score) {
      max_score = result[cluster];
    }
  }

  double sum = 0.0;

  for (StrToIntMap::iterator cit = cdict.begin(); cit != cdict.end(); ++cit) {
    std::string cluster = cit->first;
    result[cluster] = exp(result[cluster] - max_score);
    sum += result[cluster];
  }

  for (StrToIntMap::iterator cit = cdict.begin(); cit != cdict.end(); ++cit) {
    std::string cluster = cit->first;
    result[cluster] /= sum;
  }

  return result;
}


MODULE = ToyBox::XS::NBClustering		PACKAGE = ToyBox::XS::NBClustering	

NBClustering *
NBClustering::new()

void
NBClustering::DESTROY()

void
NBClustering::xs_add_instance(attributes_input)
  SV * attributes_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    StrToDoubleMap attributes;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (double)SvNV(val);
    }

    THIS->AddInstance(attributes);
  }

void
NBClustering::xs_add_labeled_instance(attributes_input, label_input)
  SV * attributes_input
  char* label_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    std::string label = std::string(label_input);
    StrToDoubleMap attributes;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (double)SvNV(val);
    }

    THIS->AddLabeledInstance(attributes, label);
  }

void
NBClustering::xs_train(cluster_num, max_iteration, epsilon, alpha, seed)
  int cluster_num
  int max_iteration
  double epsilon
  double alpha
  int seed
CODE:
  {
    THIS->Train(cluster_num, max_iteration, epsilon, alpha, seed);
  }

SV*
NBClustering::xs_predict(attributes_input)
  SV * attributes_input
CODE:
  {
    HV *hv_attributes = (HV*) SvRV(attributes_input);
    SV *val;
    char *key;
    I32 retlen;
    int num = hv_iterinit(hv_attributes);
    StrToDoubleMap attributes;
    StrToDoubleMap result;

    for (int i = 0; i < num; ++i) {
      val = hv_iternextsv(hv_attributes, &key, &retlen);
      attributes[key] = (double)SvNV(val);
    }

    result = THIS->Predict(attributes);

    HV *hv_result = newHV();
    for (StrToDoubleMap::iterator it = result.begin(); it != result.end(); ++it) {
      const char *const_key = (it->first).c_str();
      SV* val = newSVnv(it->second);
      hv_store(hv_result, const_key, strlen(const_key), val, 0); 
    }

    RETVAL = newRV_inc((SV*) hv_result);
  }
OUTPUT:
  RETVAL
  
