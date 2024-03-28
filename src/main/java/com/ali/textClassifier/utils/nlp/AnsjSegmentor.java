package com.ali.textClassifier.utils.nlp;

import com.ali.textClassifier.model.Term;
import lombok.NoArgsConstructor;
import org.ansj.splitWord.analysis.ToAnalysis;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

import static java.util.stream.Collectors.*;

@NoArgsConstructor
public class AnsjSegmentor implements NLPTools.Segmentor {

    @Override
    public List<Term> segment(String content) {
        return ToAnalysis.parse(content).getTerms().stream()
                .filter(t -> !t.getNatureStr().equals("null"))
                .map(t -> new Term(t.getName(), t.getNatureStr(), 1))
                .collect(collectingAndThen(toList(),
                        list -> {
                            list.parallelStream().collect(groupingBy(Function.identity(), counting())).forEach((term, count) -> {
                                list.stream().filter(t -> t.getWord().equals(term.getWord())).forEach(t -> t.setTf(count.intValue()));
                            });
                            return list;
                        }));
    }
}
