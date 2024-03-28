package com.ali.textClassifier.utils.nlp;

import com.ali.textClassifier.model.Term;
import com.ali.textClassifier.utils.nlp.NLPTools.Segmentor;
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.seg.Dijkstra.DijkstraSegment;
import com.hankcs.hanlp.seg.Segment;
import lombok.NoArgsConstructor;

import java.util.List;

import static java.util.stream.Collectors.toList;

@NoArgsConstructor
public class HanLPSegmentor implements Segmentor {

    private Segment segmentor = HanLP.newSegment().enableCustomDictionary(false).enableOrganizationRecognize(true);

    @Override
    public List<Term> segment(String content) {
        return segmentor.seg(content).stream().map(t -> new Term(t.word, t.nature.name(), t.getFrequency())).collect(toList());
    }
}
