import os
CACHE_PATH = "/path/to/the/huggingface/home"
os.environ["HF_HOME"] = CACHE_PATH
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_HUB_DISABLE_XET"] = "1" # the hf-xet download hangs in the vLLM engine process if the model is not cached yet

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from vllm.sampling_params import GuidedDecodingParams
from PIL import Image
from functools import lru_cache
import math

DEFAULT_BASE_SAMPLING_PARAMS = {
    "max_tokens":1024,
    "top_k":1,
}

USED_TOKEN = "[YOUR HUGGINGFACE TOKEN GOES HERE]"

def makeBatch(iterable, n=1):
    l = len(iterable)
    return [iterable[ndx:min(ndx + n, l)] for ndx in range(0, l, n)]


class VLLMModel():
    def __init__(self,modelName,revision=None,jsonTemplate=None,logProbSamplingNum=2048,hf_token=USED_TOKEN,**baseSamplingParams):
        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token # not huggingface_hub.login(): it writes the shared stored_tokens file, which parallel jobs corrupt
        self.processor = AutoProcessor.from_pretrained(
            modelName,
            revision=revision,
            download_dir=CACHE_PATH,
            use_fast=True,
            )
        self.vocab_size = self.processor.tokenizer.vocab_size
        self.logProbSamplingNum = self.vocab_size if logProbSamplingNum == -1 else logProbSamplingNum
        self.llm = LLM(
            modelName,
            revision=revision,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=32768, # prompts are short; the default (131072) does not fit the KV cache of the 27B models on one GPU
            max_logprobs=self.logProbSamplingNum,
            download_dir=CACHE_PATH,
            guided_decoding_backend="xgrammar", #important for proper guided decoding
            guided_decoding_disable_any_whitespace=True, #important for proper guided decoding
            )
        self.baseSamplingParams = DEFAULT_BASE_SAMPLING_PARAMS.copy()
        self.baseSamplingParams.update(baseSamplingParams)
        if jsonTemplate is not None: self.baseSamplingParams["guided_decoding"] = GuidedDecodingParams(json=jsonTemplate)
    
    def _toInpPrompt(self,textPrompt,imgList):
        content = []
        for img in imgList:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": textPrompt})
        messages = [{"role": "user","content": content},]
        inpPrompt = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        return inpPrompt
    
    @lru_cache
    def _encodeOneToken(self,inp):
        out = self.processor(text=inp)['input_ids'][0]
        assert len(out) == 2
        id = out[1]
        return id
    
    def _forward(self,inpText,inpImgList,jsonTemplate=None,**samplingParams):
        withImg = len(inpImgList[0]) > 0
        if withImg:
            inpBatch = [{"prompt":self._toInpPrompt(txt,imgList),"multi_modal_data":{"image":imgList}} for txt,imgList in zip(inpText,inpImgList)]
        else:
            inpBatch = [self._toInpPrompt(txt,[]) for txt in inpText]



        llmParams = self.baseSamplingParams.copy()
        llmParams.update(samplingParams)
        if jsonTemplate is not None: llmParams["guided_decoding"] = GuidedDecodingParams(json=jsonTemplate)


        outputs = self.llm.generate(inpBatch,SamplingParams(**llmParams),use_tqdm=False)
        return outputs
    
    def getLogProb(self,logProbDict,querryToken):
        if querryToken not in logProbDict: return float("-inf") # token not present => probability is very low => -inf
        return logProbDict[querryToken].logprob
    
    def getRawLogProbs(self,inpText,inpImgList,logProbQueryList,**samplingParams):
        isBatch = type(inpText) == list
        if not isBatch:
            inpText = [inpText]
            inpImgList = [inpImgList]
        
        samplingParams.update({"logprobs":self.logProbSamplingNum,"max_tokens":1})
        outputs = self._forward(inpText,inpImgList,None,**samplingParams)
        logProbs = [x.outputs[0].logprobs[0] for x in outputs]
        logProbQueryTokenList = [self._encodeOneToken(x) for x in logProbQueryList]
        logProbs = [[self.getLogProb(x,y) for y in logProbQueryTokenList] for x in logProbs]
        logProbs = [{k:v for k,v in zip(logProbQueryList,x)} for x in logProbs]
        return logProbs
    
    def getProb(self,inpText,inpImgList,logProbQueryList,decodingTemperature,**samplingParams):
        isBatch = type(inpText) == list
        if not isBatch:
            inpText = [inpText]
            inpImgList = [inpImgList]
        
        samplingParams.update({"logprobs":self.logProbSamplingNum,"max_tokens":1})
        outputs = self._forward(inpText,inpImgList,None,**samplingParams)
        logProbs = [x.outputs[0].logprobs[0] for x in outputs]
        logProbQueryTokenList = [self._encodeOneToken(x) for x in logProbQueryList]
        logProbs = [[self.getLogProb(x,y) for y in logProbQueryTokenList] for x in logProbs] # if x[y].logprob gives you an error, logProbSamplingNum was set to low => the token probhabilities of the given tokens are to high enough to be in the top-logProbSamplingNum tokens

        outProbs = list()
        outProbWeight = list()
        for sample in logProbs:
            sampleProbs = [math.exp(x/decodingTemperature) for x in sample]
            sampleSum = sum(sampleProbs)
            outProbWeight.append(sum([math.exp(x) for x in sample]))
            normProbs = [x/sampleSum for x in sampleProbs]
            probtDict = {k:v for k,v in zip(logProbQueryList,normProbs)}
            outProbs.append(probtDict)
        if not isBatch:
            outProbs = outProbs[0]
            outProbWeight = outProbWeight[0]
        return outProbs,outProbWeight

    def __call__(self, inpText,inpImgList,jsonTemplate=None,**samplingParams):
        isBatch = type(inpText) == list
        if not isBatch:
            inpText = [inpText]
            inpImgList = [inpImgList]

        outputs = self._forward(inpText,inpImgList,jsonTemplate,**samplingParams)

        textOutputs = [x.outputs[0].text.strip() for x in outputs]
        if not isBatch: textOutputs = textOutputs[0]
        return textOutputs