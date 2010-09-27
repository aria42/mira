(ns mira  
  {:doc "Implements margin-infused relaxation algorithm (MIRA). Fairly optimized."
   :author "Aria Haghighi <me@aria42.com>"}
  (:gen-class)
  (:use [clojure.string :only [join]]
        [clojure.java.io :only [reader]]))
                      
(defn dot-product 
  "dot-product between two maps (sum over matching values)
   Bottleneck: written to be efficient"
  [x y]  
  (loop [sum 0.0 y y]
    (let [f (first y)]
      (if-not f sum        
        (let [k (first f)  v (second f)]
          (recur (+ sum (* (get x k 0.0) v))
                 (rest y)))))))

(defn sum [f xs]
  (reduce + (map f xs)))

(defn norm-sq 
  "||x||^2 over values in map x"
  [x] (sum #(* % %) (map second x)))
                   
(defn add-scaled 
 "x <- x + scale * y
  Bottleneck: written to be efficient"
 [x scale y]
 (persistent!
  (reduce
    (fn [res elem]
      (let [k (first elem) v (second elem)]
         (assoc! res k (+ (get x k 0.0) (* scale v)))))
     (transient x)
     y)))
     
; Needed for averaged weight vector
(def +updates-left+ (atom nil))   
           
; (cum)-label-weights: label -> (cum)-weights
(defrecord Mira [loss-fn label-weights cum-label-weights])   

(defn new-mira 
  [labels loss-fn]
  (let [empty-weights #(into {} (for [l labels] [l {}]))]
    (Mira. loss-fn (empty-weights) (empty-weights))))          
     
(defn get-labels 
  "return possible labels for task"
  [mira]  (keys (:label-weights mira)))  
  
(defn get-score-fn 
  "return fn: label => model-score-of-label"
  [mira datum]
  (fn [label] 
    (dot-product ((:label-weights mira) label) datum)))

(defn get-loss 
  "get loss for predicting predict-label in place of gold-label"
  [mira gold-label predict-label]
  ((:loss-fn mira) gold-label predict-label))        

(defn ppredict
   "When you have lots of classes,  useful to parallelize prediction"
  [mira datum]
  (let [score-fn (get-score-fn mira datum)        
        label-parts (partition-all 5 (get-labels mira))
        part-fn (fn [label-part]
                  (reduce 
                    (fn [res label] (assoc res label (score-fn label)))
                    {} label-part))
        score-parts (pmap part-fn label-parts)
        scores (apply merge score-parts)]
    (first (apply max-key second scores))))
               
(defn predict 
  "predict highest scoring class"
  [mira datum]
  (if (> (count (get-labels mira)) 5)
    (ppredict mira datum)
    (apply max-key (get-score-fn mira datum) (get-labels mira))))
    
(defn update-weights
  "returns new weights assuming error predict-label instead of gold-label.
   delta-vec is the direction and alpha the scaling constant"
  [label-weights delta-vec gold-label predict-label alpha]  
  (->  label-weights
       (update-in [gold-label]  add-scaled alpha delta-vec)          
       (update-in [predict-label] add-scaled (- alpha) delta-vec)))             

(defn update-mira
  "update mira for an example returning [new-mira error?]"
  [mira datum gold-label]
  (let [predict-label (ppredict mira datum)]
       (if (= predict-label gold-label)
            ; If we get it right do nothing
            [mira false]
            ; otherwise, update weights
            (let [score-fn (get-score-fn mira datum)
                  loss (get-loss mira gold-label predict-label)
                  gap (- (score-fn gold-label) (score-fn predict-label))
                  alpha  (/ (- loss  gap) (* 2 (norm-sq datum)))
                  avg-factor (* @+updates-left+ alpha)
                  new-mira (-> mira 
                            ; Update Current Weights
                            (update-in [:label-weights]
                              update-weights datum gold-label predict-label alpha)
                            ; Update Average (cumulative) Weights  
                            (update-in [:cum-label-weights]
                              update-weights datum gold-label 
                              predict-label avg-factor))]
              [new-mira true]))))
          
(defn train-iter 
  "Training pass over data, returning [new-mira num-errors], where
   num-errors is the number of mistakes made on training pass"
  [mira labeled-data-fn]
   (reduce
     (fn [[cur-mira num-errors] [datum gold-label]]
       (let [[new-mira error?] (update-mira cur-mira datum gold-label)]          
          (swap! +updates-left+ dec)
          [new-mira (if error? (inc num-errors) num-errors)]))
     [mira 0]
     (labeled-data-fn)))
    
(defn train 
  "do num-iters iterations over labeled-data (yielded by labeled-data-fn)"
  [labeled-data-fn labels num-iters loss-fn]
  (loop [iter 0 mira (new-mira labels loss-fn)]        
    (if (= iter num-iters) 
        mira
        (let [[new-mira num-errors]  (train-iter mira labeled-data-fn)]          
          (println 
            (format "[MIRA] On iter %s made %s training mistakes" iter num-errors))                  
          ; If we don't make mistakes, never will again  
          (if (zero? num-errors) 
            new-mira (recur (inc iter) new-mira))))))

(defn feat-vec-from-line 
  "format: feat1:val1 ... featn:valn. feat is a string and val a double"
  [#^String line]
  (for [#^String piece (.split line "\\s+")
        :let [split-index (.indexOf piece ":")
              feat (if (neg? split-index)                 
                      piece
                      (.substring piece 0 split-index))
              value (if (neg? split-index) 1
                      (-> piece (.substring (inc split-index)) 
                          Double/parseDouble))]]
    [feat value]))
                        
(defn load-labeled-data 
  "format: label feat1:val1 .... featn:valn"
  [path]
  (for [line (line-seq (reader path))
        :let [pieces (.split #^String line "\\s+")
              label (first pieces)
              feat-vec (feat-vec-from-line (join " " (rest pieces)))]]
    [feat-vec label]))              

(defn load-data 
  "load data without label"
  [path] (map feat-vec-from-line (line-seq (reader path))))
  
(defn normalize-vec [x]
  (let [norm (Math/sqrt (norm-sq x))]
    (into {} (for [[k v] x] [k (/ v norm)]))))
            
(defn -main [& args]
  (case (first args)
    "train"
      (let [[data-path num-iters outfile] (rest args)
            labeled-data-fn #(load-labeled-data data-path)
            labels (into #{} (map second (labeled-data-fn)))
            num-iters (Integer/parseInt num-iters)]
        ; For Average Weight Calculation
        (compare-and-set! +updates-left+ nil (* num-iters (count (labeled-data-fn))))
        (let [mira (train labeled-data-fn labels num-iters  (constantly 1))
              avg-weights (into {}                
                            (for [[label sum-weights] (:cum-label-weights mira)]
                              [label (normalize-vec sum-weights)]))]
          (println "[MIRA] Done Training. Writing weights to " outfile)
          (spit outfile avg-weights)))
    "predict"
      (let [[weight-file data-file] (rest args)
            weights (read-string (slurp weight-file))
            mira (Mira.  (constantly 1) weights weights)]
        (doseq [datum (load-data data-file)]
          (println (predict mira datum))))
    "test"
      (let [[weight-file data-file] (rest args)
            weights (read-string (slurp weight-file))
            mira (Mira. (constantly 1) weights weights)
            labeled-test (load-labeled-data data-file)
            gold-labels (map second labeled-test)
            predict-labels (map #(predict mira %) (map first labeled-test))
            num-errors (->> (map vector gold-labels predict-labels)
                            (sum (fn [[gold predict]] (if (not= gold predict) 1 0))))]
        (println "Error: " (double (/ num-errors (count gold-labels)))))))        