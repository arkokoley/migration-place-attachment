extensions [table]

globals [map_spread_factor max_color_spread layer_functions max_mean_q max_sd_q max_mean_i max_sd_i max_a
  max_group_diff
  place_list
  distance_table
  far_from_home
  at_home moves
  avg_move_distance     ; Average distance of moves
  yearly_moves         ; Number of moves in the current year
  total_income_change  ; Track total income changes from moves
  total_qol_change    ; Track total quality of life changes from moves
  cyclical_moves      ; Count of moves that return to a previous location
  time_since_moves    ; List to track time between moves

  ; For yearly tracking
  year_start_tick    ; Track when each year starts
  move_history       ; Table to store movement history
]

breed [people person]
breed [places place]

patches-own [location state_name in_context]
people-own [home_state home_state_name resident_state_name class attachment_level attachment preferences observations layer_observations pInitiate decisionType satisfied? uncertain? target_place]
places-own [name place_location place_mean_q place_sd_q place_a place_mean_i place_sd_i admin_level]


to setup

  clear-all
  reset-ticks

  ;set globals
  set map_spread_factor 20
  set max_color_spread 100
  set-default-shape places "blank"

  ;;bounds for place parameters
  set max_mean_q 50
  set max_sd_q 10
  set max_mean_i 50
  set max_sd_i 50
  set max_a initial_population * 10
  set max_group_diff 0.2

  ;;define layer functions - anonymous functions of place qualities that take place & agent-specific inputs as parameters
  set layer_functions table:make
  table:put layer_functions "quality_life" [[mean_q sd_q] -> random-normal mean_q sd_q]
  table:put layer_functions "income" [[mean_i sd_i n_agents a agent_class] -> (random-normal (agent_class * mean_i) sd_i) / (1 + n_agents / a)]

  setup-map

  setup-agents

  setup-network



  ;;define place parameters
  foreach n-values (max [item 0 location] of places) [i -> i + 1] [i ->
      let start_place_mean_q random max_mean_q
      let start_place_sd_q random max_sd_q
      let start_place_mean_i random max_mean_i
      let start_place_sd_i random max_sd_i
      let start_place_a random max_a + 1

    ask places with [item 0 location = i] [
      set place_mean_q start_place_mean_q
      set place_sd_q start_place_sd_q
      set place_mean_i start_place_mean_i
      set place_sd_i start_place_sd_i
      set place_a start_place_a
    ]
  ]


  foreach n-values (admin_levels - 1) [i -> i + 1] [ j ->
    let current_unique_states remove-duplicates [sublist location 0 j] of places
    foreach current_unique_states [current_state ->
      let delta_pmq ((random-float 1) - 0.5) * max_group_diff
      let delta_psq ((random-float 1) - 0.5) * max_group_diff
      let delta_pmi ((random-float 1) - 0.5) * max_group_diff
      let delta_psi ((random-float 1) - 0.5) * max_group_diff
      let delta_a ((random-float 1) - 0.5) * max_group_diff


      ask places with [sublist location 0 j = current_state] [
        set place_mean_q place_mean_q * (1 + delta_pmq)
        set place_sd_q place_sd_q * (1 + delta_psq)
        set place_mean_i place_mean_i * (1 + delta_pmi)
        set place_sd_i place_sd_i * (1 + delta_psi)
        set place_a place_a * (1 + delta_a)
      ]
    ]

  ]

  setup-migration-tracking

end

to go

  ;;update any time functions
  ask people [update-place-attachment]

  reset-metrics

  ;; let people do all the things they do
  ask people [get-utility]
  ask people [interact]
  ask people [decide]
  ask people [move]

  compute-metrics
  update-migration-metrics

  tick


end

to-report agent-data-csv
  file-open "agent_data.csv"
  file-print "who,resident_state_name"
  ask people [
    file-print (word who ", " resident_state_name)
  ]
  file-close
  report "Data exported to agent_data.csv"
end

to update-place-attachment

  ;;increment place attachment where the agent is (but don't add it back in yet)
  let current_place table:get attachment resident_state_name
  set current_place min (list 1 (current_place + attachment_form_rate))

  ;;decrement place attachment where the agent isn't
  let place_names table:keys attachment

  foreach place_names [x ->
    let place_attachment table:get attachment x
    table:put attachment x max (list 0 (place_attachment - attachment_decay_rate))
  ]

  ;;now add in the previously calculated value for place attachment
  table:put attachment resident_state_name current_place

end

to-report calculate-utility [quality_of_life income where_i_am]
  ;; estimate utility as the sum across layers, multiplied by a function of place attachment, and scaled by attachment to other places.
  ;; could improve with a distance measure
  let utility 0
  set utility (item 0 preferences) * quality_of_life + (item 1 preferences * income)
  set utility utility * (1 + attachment_level * (table:get attachment where_i_am))
  ;calculate the attachment-distance product
  let attachment_distance_product 0
  foreach place_list [ i ->
    set attachment_distance_product attachment_distance_product + (table:get attachment i) * (table:get table:get distance_table where_i_am i)
  ]
  set utility utility / (1 + attachment_distance_product)
  report utility
end

to get-utility

  let where_i_am [state_name] of patch-here

  let quality_of_life 0
  let income 0
  let my_class class
  ask one-of places with [name = where_i_am] [
    set quality_of_life (runresult (table:get layer_functions "quality_life") place_mean_q place_sd_q)
    set income (runresult (table:get layer_functions "income") place_mean_i place_sd_i (count people with [resident_state_name = where_i_am]) place_a my_class)
  ]
  let utility calculate-utility quality_of_life income where_i_am

  add-layer-observation 0 quality_of_life where_i_am
  add-layer-observation 1 income where_i_am
  add-observation utility where_i_am
end

to estimate-utility-from-layer-observation [where_i_am]

  let quality_of_life item 0 item 0 (table:get layer_observations where_i_am)
  let income item 0 item 1 (table:get layer_observations where_i_am)
  let utility calculate-utility quality_of_life income where_i_am
  add-observation utility where_i_am
end

to add-layer-observation [ index layer_value current_place]

  let layer_observation_list table:get layer_observations current_place
  let current_layer item index layer_observation_list
  set current_layer fput layer_value current_layer
  if length current_layer > max_memory [
    set current_layer sublist current_layer 0 (max_memory - 1)
  ]
  set layer_observation_list replace-item index layer_observation_list current_layer
  table:put layer_observations current_place layer_observation_list

end

to add-observation [ utility current_place]

  let observation_list table:get observations current_place
  set observation_list fput utility observation_list
  if length observation_list > max_memory [
    set observation_list sublist observation_list 0 (max_memory - 1)
  ]
  table:put observations current_place observation_list

end

to interact

  ;;set the number of possible interactions
  let numInteractions maxInitiateInteractions
  while [numInteractions > 0] [

    ;;see if this interaction happens
    if random-float 1 < pInitiate [;; have an interaction with someone!

      ;;call a friend
      if (any? link-neighbors) [
        let partner one-of link-neighbors

        ;;draw the number of bits of info shared to
        let numPiecesTo random max_nInformation
        while [numPiecesTo > 0] [

          let topic one-of place_list
          let layer_obs_topic table:get layer_observations topic
          let length_topic length item 0 layer_obs_topic
          if length_topic > 0 [ ;; i.e., if there are values in any of the layer function outputs stored
            let topic_item random length_topic
            foreach n-values (table:length layer_functions) [i -> i] [ i ->
              let current_item item topic_item (item i layer_obs_topic)
              ask partner [add-layer-observation i current_item topic]
            ]
            ;;estimate the utility from the layers we just added, which we know are in the first point
            ask partner [estimate-utility-from-layer-observation topic]
          ]

          set numPiecesTo numPiecesTo - 1
        ]
        ask partner [

          ;;draw the number of bits of info shared from
          let numPiecesFrom random max_nInformation
          while [numPiecesFrom > 0] [

            let topic one-of place_list
            let layer_obs_topic table:get layer_observations topic
            let length_topic length item 0 layer_obs_topic
            if length_topic > 0 [ ;; i.e., if there are values in any of the layer function outputs stored
              let topic_item random length_topic
              foreach n-values (table:length layer_functions) [i -> i] [ i ->
                let current_item item topic_item (item i layer_obs_topic)
                ask myself [add-layer-observation i current_item topic]

              ]
            ]

            set numPiecesFrom numPiecesFrom - 1
          ]

        ]

      ]


    ]

    set numInteractions numInteractions - 1
  ]

end

to decide
  if switchInterval > 0 and ticks mod switchInterval = 0 [
    set decisionType (ifelse-value (random-float 1 < pConsumat) [ "consumat"] [ "maximize"])
  ]

  (ifelse
    decisionType = "consumat" [
      decide-consumat
    ]
    decisionType = "maximize" [
      decide-maximize
    ])


end

to setup-agents

  set-default-shape people "person"

  ;;set up the blank observation and attachment tables to put into agents
  let attachment_table place_list
  let observation_table place_list
  let layer_observation_table place_list
  foreach n-values (length attachment_table) [i -> i] [i ->
    let current_item item i attachment_table
    let current_item_1 (list current_item 0)
    let current_item_2 (list current_item [])
    let current_item_3 (list current_item n-values (table:length layer_functions) [[]])
    set attachment_table replace-item i attachment_table current_item_1
    set observation_table replace-item i observation_table current_item_2
    set layer_observation_table replace-item i layer_observation_table current_item_3
  ]
  set attachment_table table:from-list attachment_table
  set observation_table table:from-list observation_table
  set layer_observation_table table:from-list layer_observation_table

  create-people initial_population [
   setxy random-xcor random-ycor
    set home_state [location] of patch-here
    set home_state_name reduce [[a b] -> (word a "_" b)] location
    set resident_state_name home_state_name
    set attachment copy-table attachment_table
    table:put attachment home_state_name initial_attachment
    set attachment_level (max (list 0.01 (min (list 1 random-normal mean_attachment sd_attachment))))
    set pInitiate (max (list 0 (min (list 1 random-normal mean_pInitiate sd_pInitiate))))
    set preferences n-values (table:length layer_functions) [(max (list 0.01 (min (list 1 random-normal mean_pref sd_pref))))]
    set observations copy-table observation_table
    set layer_observations copy-table layer_observation_table
    set decisionType (ifelse-value (random-float 1 < pConsumat) [ "consumat"] [ "maximize"])
    set target_place resident_state_name

  ]

end

to setup-network

  ;clear any previous network
  ask links [die]

  ;rescale probabilities to sum to one
  let total_p (p_link_state + p_link_random + p_link_network)
  set p_link_state p_link_state / total_p
  set p_link_random p_link_random / total_p
  set p_link_network p_link_network / total_p

  let n_links ave_links_person * count people


  let p_1 p_link_state
  let p_2 p_link_state + p_link_network

  while [n_links > 0] [

    let link_handled 0

    let person_1 one-of people

    ask person_1 [

      let r_draw random-float 1

      if (r_draw < p_1 and link_handled = 0) [

       ;;assign link based on state
        let others_in_state other people with [home_state_name = [home_state_name] of myself]
        if any? others_in_state [
          create-link-with one-of others_in_state
          set n_links n_links - 1
        ]
        set link_handled 1
      ]

      if (r_draw < p_2 and link_handled = 0) [
        ;;choose based on network
        let friends link-neighbors
        let other_friends friends
        ask friends [
         set other_friends (turtle-set other_friends link-neighbors)
        ]

        if any? other other_friends [
         create-link-with one-of other other_friends
         set n_links n_links - 1
        ]

       set link_handled 1
      ]

      if (link_handled = 0) [
       create-link-with one-of other people
       set n_links n_links - 1

      ]


    ]

  ]

end

to setup-map


  ;initialize locations
  ask patches [set location n-values admin_levels [0]]

  let place_todo []
  let place_level []
  let current_state_name reduce [[a b] -> (word a "_" b)] [location] of one-of patches
  ask patches [set state_name current_state_name]
  set place_todo lput  current_state_name place_todo
  set place_level lput 0 place_level


  while [length place_todo > 0][


    let current_place item 0 place_todo
    let current_level item 0 place_level
    set place_todo but-first place_todo
    set place_level but-first place_level
    ask patches [set in_context 0]
    ask patches with [state_name = current_place] [set in_context 1]
    let current_context count patches with [in_context = 1]

    ;figure out how many parts we have for this place
    let num_parts max (list (round random-normal mean_parts sd_parts) 1)


    let current_part 1

    ;seeding our parts

    ask n-of (min (list current_context num_parts)) patches with [in_context = 1] [

      set location replace-item current_level location current_part
      set current_state_name reduce [[a b] -> (word a "_" b)] location ;; (sublist location 0 (current_level + 1))
      set state_name current_state_name
      if (current_level < admin_levels - 1) [
        set place_todo lput  current_state_name place_todo
        set place_level lput  (current_level + 1) place_level
      ]

      set current_part current_part + 1

      ;;think of each of these seeds like a 'capital' that we'll define as a place agent
      sprout-places 1 [
        set admin_level current_level
        set name (reduce [[a b] -> (word a "_" b)] location)
        set place_location location
      ]
    ]



    ;let the parts fill out
    while [any? patches with [in_context = 1 and item current_level location = 0]] [


      ask patches with [in_context = 1 and item current_level location = 0] [
        let assigned_neighbors neighbors4 with [in_context = 1 and item current_level location > 0]

        if any? assigned_neighbors [
          set current_part [item current_level location] of one-of assigned_neighbors
          set location replace-item current_level location current_part
          set current_state_name reduce [[a b] -> (word a "_" b)] location
          set state_name current_state_name
        ]
      ]

    ];;end fill-out while loop

  ];; end subdivide new place while loop

  ;;make a list of all unique places, now that all patches assigned
  set place_list remove-duplicates [name] of places

  ;;calculate distance matrix
  let max_distance 0
  set distance_table table:make
  let distance_column place_list
  foreach n-values (length distance_column) [i -> i] [i ->
    let current_item item i distance_column
    let current_item_1 (list current_item 0)
    set distance_column replace-item i distance_column current_item_1
  ]
  set distance_column table:from-list distance_column
  foreach place_list [ i ->
    let current_column copy-table distance_column
    ask one-of places with [name = i] [
      foreach place_list [ j ->
        let current_distance distance one-of places with [name = j]
        table:put current_column j current_distance
        if current_distance > max_distance [
         set max_distance current_distance
        ]
      ]
    ]
    table:put distance_table i current_column
  ]

  ;;normalize the whole table to have a max value of 1
  foreach place_list [ i ->
    let current_column table:get distance_table i
    foreach place_list [ j ->
      let current_distance table:get current_column j
      table:put current_column j (current_distance / max_distance)
    ]
  ]


  ;color the patches
  ask patches [set pcolor ((reduce [[a b] -> map_spread_factor * a + b] location) / (10 ^ (length location)) * max_color_spread)]

end

to-report copy-table [ orig ]
  let copy table:make
  foreach ( table:keys orig ) [
    [key] -> table:put copy key ( table:get orig key )
  ]
  report copy
end

to decide-consumat

    define-satisfaction-certainty

  ifelse satisfied? [
    ifelse uncertain? [
      decide-look-to-peers
    ]
    [
      decide-stay
    ]
  ]
  [
    ifelse uncertain? [
      decide-social-comparison
    ]
    [
      decide-deliberate-move
    ]
  ]


end


to decide-maximize
    let where_i_am [state_name] of patch-here
  let current_utility_list table:get observations where_i_am
  let current_utility 0
  if not empty? current_utility_list [ set current_utility mean current_utility_list ]
  let best where_i_am

  let memory table:keys observations
  foreach memory [
    plc ->
      let utility_list table:get observations plc
      let utility 0
      if not empty? utility_list [ set utility mean utility_list ]
      if utility > current_utility [
        set best plc
        set current_utility utility
      ]
  ]

  set target_place best

end


to-report get-weighted-average [l scale]
  if length l = 0 [ report 0 ]
  let current_modifier 1
  let total 0
  foreach l [ [i] -> set total total + (current_modifier * i) set current_modifier current_modifier * scale ]
  set total total / length l

  report total
end

to-report all-place-observation-weighted-average [scale]
  let num_places length place_list
  let averages n-values num_places [i -> get-weighted-average table:get observations item i place_list scale ]
  report averages
end

to-report get-satisfaction
  let observation_list table:get observations resident_state_name
  let scale 0.5
  let observation_total get-weighted-average observation_list scale

  report observation_total
end

to-report get-uncertainty
  let observation_list table:get observations resident_state_name
  if length observation_list < 2 [ report 0 ]
  let observation_sd standard-deviation observation_list
  let observation_mean mean observation_list

  report observation_sd
end

to define-satisfaction-certainty
  let average_neighbor_satisfaction 0
  if any? link-neighbors [
    ask link-neighbors [
      let my-satisfaction get-satisfaction
      set average_neighbor_satisfaction average_neighbor_satisfaction + my-satisfaction
    ]

  set average_neighbor_satisfaction average_neighbor_satisfaction / count link-neighbors
  let my-satisfaction get-satisfaction

  set satisfied? TRUE
  if my-satisfaction < 0.5 * average_neighbor_satisfaction [ set satisfied? FALSE ]


  let average_neighbor_uncertainty_sd 0
  ask link-neighbors [
    let my-uncertainty get-uncertainty
    set average_neighbor_uncertainty_sd average_neighbor_uncertainty_sd + my-uncertainty
  ]
  set average_neighbor_uncertainty_sd average_neighbor_uncertainty_sd / count link-neighbors
  let my-uncertainty-sd get-uncertainty

  set uncertain? TRUE
  if my-uncertainty-sd < 1.5 * average_neighbor_uncertainty_sd [ set uncertain? FALSE ]
  ]

  ; JUST FOR NOW!!!!!!!!!!!!
  set satisfied? get-satisfaction > 0
  set uncertain? FALSE

end


to decide-stay
end

to decide-look-to-peers ; look at people in a similar position (same region? same satisfaction/utility?)
end

to decide-deliberate-move
  let place_scores all-place-observation-weighted-average 0.5
  let max_i position max place_scores place_scores
  let max_place item max_i place_list
  set target_place max_place ; probably want to make this a probability rather than just setting a target place
end

to decide-social-comparison ; look at friends
end

to move

  (ifelse moveMethod = "patch" [
    if resident_state_name != target_place [
      face one-of places with [name = [target_place] of myself]
      forward 1
      set resident_state_name [state_name] of patch-here
      set moves moves + 1
    ]

    ]
    moveMethod = "place" [
      let myTarget target_place
      if resident_state_name != myTarget and any? patches with [state_name = myTarget] [
        set resident_state_name myTarget
        move-to one-of patches with [state_name = myTarget]
        set moves moves + 1
      ]
  ])


end


to reset-metrics
  set moves 0
  set far_from_home 0
  set at_home 0
end

to compute-metrics

  ;; moves is updated at agent level

  ;; distances from home
  ask people [
    let myHome home_state_name
    set far_from_home far_from_home + distance (one-of places with [name = myHome])]
  set far_from_home (far_from_home / (count people))


  ;; people at home
  set at_home count people with [resident_state_name = home_state_name]

end

to setup-migration-tracking
  set yearly_moves 0
  set year_start_tick 0
  set move_history table:make
  ask people [
    table:put move_history who (list resident_state_name)
  ]
end

to update-migration-metrics
  ; Update yearly moves if we've completed a year (12 ticks)
  if ticks mod 12 = 0 [
    set yearly_moves moves
    set year_start_tick ticks
  ]

  ; Track movement history and calculate metrics
  ask people [
    let current_location resident_state_name
    let history table:get move_history who

    ; If location changed, update metrics
    if current_location != last history [
      ; Add new location to history
      table:put move_history who (lput current_location history)

      ; Calculate and update metrics
      update-move-metrics current_location history
    ]
  ]
end

to update-move-metrics [current_location history]
  ; Calculate move distance
  let prev_location last history
  let move_dist distance one-of places with [name = prev_location]
  set avg_move_distance (avg_move_distance * (moves - 1) + move_dist) / moves

  ; Check for cyclical moves
  if length history >= 2 and member? current_location but-last history [
    set cyclical_moves cyclical_moves + 1
  ]

  ; Track quality of life and income changes
  let old_qol mean [place_mean_q] of places with [name = prev_location]
  let new_qol mean [place_mean_q] of places with [name = current_location]
  let old_income mean [place_mean_i] of places with [name = prev_location]
  let new_income mean [place_mean_i] of places with [name = current_location]

  set total_qol_change total_qol_change + (new_qol - old_qol)
  set total_income_change total_income_change + (new_income - old_income)
end

to-report get-avg-improvement
  report (list
    ifelse-value moves > 0 [total_income_change / moves][0]
    ifelse-value moves > 0 [total_qol_change / moves][0]
  )
end
@#$#@#$#@
GRAPHICS-WINDOW
210
10
647
448
-1
-1
13.0
1
10
1
1
1
0
0
0
1
-16
16
-16
16
0
0
1
ticks
30.0

BUTTON
14
414
80
448
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
18
26
190
59
admin_levels
admin_levels
1
3
3.0
1
1
NIL
HORIZONTAL

SLIDER
17
70
189
103
mean_parts
mean_parts
1
5
3.0
1
1
NIL
HORIZONTAL

SLIDER
18
115
190
148
sd_parts
sd_parts
0
3
1.0
1
1
NIL
HORIZONTAL

SLIDER
17
162
189
195
initial_population
initial_population
1
1000
500.0
1
1
NIL
HORIZONTAL

SLIDER
16
214
189
247
ave_links_person
ave_links_person
0
6
1.0
1
1
NIL
HORIZONTAL

SLIDER
16
267
189
300
p_link_state
p_link_state
0
1
0.09909909909909911
0.01
1
NIL
HORIZONTAL

SLIDER
16
312
189
345
p_link_network
p_link_network
0
1
0.4701649056191167
.01
1
NIL
HORIZONTAL

SLIDER
16
358
189
391
p_link_random
p_link_random
0
1
0.43073599528178425
0.01
1
NIL
HORIZONTAL

SLIDER
690
25
862
58
num_classes
num_classes
1
5
3.0
1
1
NIL
HORIZONTAL

SLIDER
688
72
878
105
attachment_form_rate
attachment_form_rate
0
0.1
0.019
0.001
1
NIL
HORIZONTAL

SLIDER
687
122
883
155
attachment_decay_rate
attachment_decay_rate
0
0.1
0.02
0.001
1
NIL
HORIZONTAL

SLIDER
689
175
861
208
initial_attachment
initial_attachment
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
689
314
861
347
mean_pref
mean_pref
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
688
363
860
396
sd_pref
sd_pref
0
1
0.2
0.01
1
NIL
HORIZONTAL

SLIDER
688
413
860
446
max_memory
max_memory
0
20
10.0
1
1
NIL
HORIZONTAL

BUTTON
111
415
174
448
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

SLIDER
917
317
1089
350
mean_attachment
mean_attachment
0
1
0.0
0.01
1
NIL
HORIZONTAL

SLIDER
916
367
1088
400
sd_attachment
sd_attachment
0
1
0.5
0.01
1
NIL
HORIZONTAL

SLIDER
921
24
1116
57
maxInitiateInteractions
maxInitiateInteractions
0
5
2.0
1
1
NIL
HORIZONTAL

SLIDER
929
91
1101
124
mean_pInitiate
mean_pInitiate
0
1
0.3
0.01
1
NIL
HORIZONTAL

SLIDER
930
135
1102
168
sd_pInitiate
sd_pInitiate
0
1
0.1
0.01
1
NIL
HORIZONTAL

SLIDER
930
180
1102
213
max_nInformation
max_nInformation
0
10
5.0
1
1
NIL
HORIZONTAL

SLIDER
1235
30
1407
63
pConsumat
pConsumat
0
1
0.42
0.01
1
NIL
HORIZONTAL

INPUTBOX
1237
93
1386
153
switchInterval
5.0
1
0
Number

TEXTBOX
1255
161
1405
179
(Use 0 for 'never switch')\n
11
0.0
1

CHOOSER
1243
223
1381
268
moveMethod
moveMethod
"patch" "place"
1

PLOT
54
503
254
653
Average distance of agents from home (centroid)
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot far_from_home"

PLOT
282
504
482
654
Number of moves
NIL
NIL
0.0
10.0
0.0
10.0
true
false
"" ""
PENS
"default" 1.0 0 -16777216 true "" "plot moves"

@#$#@#$#@
## WHAT IS IT?

(a general understanding of what the model is trying to show or explain)

## HOW IT WORKS

(what rules the agents use to create the overall behavior of the model)

## HOW TO USE IT

(how to use the model, including a description of each of the items in the Interface tab)

## THINGS TO NOTICE

(suggested things for the user to notice while running the model)

## THINGS TO TRY

(suggested things for the user to try to do (move sliders, switches, etc.) with the model)

## EXTENDING THE MODEL

(suggested things to add or change in the Code tab to make the model more complicated, detailed, accurate, etc.)

## NETLOGO FEATURES

(interesting or unusual features of NetLogo that the model uses, particularly in the Code tab; or where workarounds were needed for missing features)

## RELATED MODELS

(models in the NetLogo Models Library and elsewhere which are of related interest)

## CREDITS AND REFERENCES

(a reference to the model's URL on the web if it has one, as well as any other necessary credits, citations, and links)
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

blank
true
0

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
