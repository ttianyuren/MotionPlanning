(define (domain pick-and-place)
  (:requirements :strips :equality :action-costs)
  (:types wuti grasp_dir grasp config pose trajectory)
  (:predicates

    (issink ?r - wuti)
    (isstove ?r - wuti)
    (fixed ?r - wuti)
    (graspable ?o - wuti)

    (IsGraspDir ?o - wuti ?p - pose ?dg - grasp_dir)
    (isgrasp ?o - wuti ?g - grasp)
    (GraspAtPose ?g - grasp ?p - pose)


    (stackable ?o - wuti ?r - wuti)
    (ispose ?o - wuti ?p - pose)
    (isconf ?q - config)
    (istraj ?t - trajectory)
    (iskin ?o - wuti ?p - pose ?g - grasp ?q - config ?t - trajectory)
    (isfreemotion ?q1 - config ?t - trajectory ?q2 - config)
    (isholdingmotion ?q1 - config ?t - trajectory ?q2 - config ?o - wuti ?g - grasp)
    (trajcollision ?t - trajectory ?o2 - wuti ?p2 - pose)

    (issupport ?o - wuti ?p - pose ?r - wuti)
    (isstacksupport ?ou - wuti ?pu - pose ?os - wuti ?ps - pose)


    (atpose ?o - wuti ?p - pose)
    (atgrasp ?o - wuti ?g - grasp)
    (handempty)
    (atconf ?q - config)
    (canmove)
    (cleaned ?o - wuti)
    (cooked ?o - wuti)

    (on ?o - wuti ?r - wuti)
    (occupied ?o - wuti)
    (holding ?o - wuti)

    (usedgrasp ?g)
    (canpick)
    (canplace)

    (IsMeasuredPose ?o - wuti ?p - pose)
    (located ?o - wuti ?p - pose)
    (allowLocate)
    (issensor ?o - wuti)

    (usedGrasp ?o - wuti ?p - pose ?g - grasp)
  )

  (:functions
      (total-cost) - number
  )


  (:action locate_body
    :parameters (?o - wuti  ?p - pose ?os - wuti)
    :precondition (and (graspable ?o)
                        (allowLocate)
                        (issensor ?os)
                        (canmove)
                       )
    :effect (and (located ?o ?p)
                 (not (allowLocate))
                 (increase (total-cost) 50)
                 )
  )

  (:action move_free
    :parameters (?q1 - config ?q2 - config ?t - trajectory)
    :precondition (and (isfreemotion ?q1 ?t ?q2)
                       (atconf ?q1) (handempty) (canmove))
    :effect (and (atconf ?q2)
                 (not (atconf ?q1))
                 (not (canmove))
                 (canpick)
                 (increase (total-cost) 100)
                 )
  )
  (:action move_holding
    :parameters (?q1 - config ?q2 - config ?o - wuti ?g - grasp ?t - trajectory)
    :precondition (and (isholdingmotion ?q1 ?t ?q2 ?o ?g)
                       (atconf ?q1) (atgrasp ?o ?g) (canmove))
    :effect (and (atconf ?q2)
                 (not (atconf ?q1))
                 (not (canmove))
                 (canpick)
                 (canplace)
                 (increase (total-cost) 100)
                 )
  )

  (:action pick
    :parameters (?o - wuti ?p - pose ?g - grasp ?q - config ?t - trajectory)
    :precondition (and (iskin ?o ?p ?g ?q ?t)
                       (atpose ?o ?p) (handempty) (atconf ?q)
                       (canpick)
                       (located ?o ?p)
                       (not (usedGrasp ?o ?p ?g))
                       (GraspAtPose ?g ?p)
                       )
    :effect (and (atgrasp ?o ?g)
                 (canmove)
                 (not (atpose ?o ?p))
                 (not (handempty))
                 (increase (total-cost) 100)

                 )
  )

  ;(:action diffpick
  ;  :parameters (?o - wuti ?p - pose ?g - grasp ?q - config ?t - trajectory)
  ;  :precondition (and (iskin ?o ?p ?g ?q ?t)
  ;                     (atpose ?o ?p) (handempty) (atconf ?q)
  ;                     (not (unsafetraj ?t))
  ;                     (canpick)
  ;                     (pickonce ?o)
  ;                     (located ?o ?p)
  ;                     )
  ;  :effect (and (atgrasp ?o ?g)
  ;               (canmove)
  ;               (not (atpose ?o ?p))
  ;               (not (handempty))
  ;               (increase (total-cost) 120)
  ;
  ;               )
  ;)

  (:action place
    :parameters (?o - wuti ?p - pose ?r - wuti ?g - grasp ?q - config ?t - trajectory)
    :precondition (and (iskin ?o ?p ?g ?q ?t)
                       (issupport ?o ?p ?r)
                       (atgrasp ?o ?g)
                       (atconf ?q)
                       (graspable ?o)
                       (fixed ?r)
                       (canplace)
                       )
    :effect (and (atpose ?o ?p)
                 (handempty)
                 (canmove)
                 (not (atgrasp ?o ?g))
                 (not (canpick))
                 (not (canplace))
                 (increase (total-cost) 100)
                 (allowLocate)
                 (not (located ?o ?p))
                 (usedGrasp ?o ?p ?g)
                 )
  )
  (:action stack
    :parameters (?ou - wuti ?pu - pose ?os - wuti ?ps - pose ?g - grasp ?q - config ?t - trajectory)
    :precondition (and (atpose ?os ?ps)
                       (isstacksupport ?ou ?pu ?os ?ps)
                       (iskin ?ou ?pu ?g ?q ?t)
                       (atgrasp ?ou ?g)
                       (atconf ?q)
                       (graspable ?ou)
                       (graspable ?os)
                       )
    :effect (and (atpose ?ou ?pu)
                 (handempty)
                 (canmove)
                 (not (atgrasp ?o ?g))
                 (not (canpick))
                 (increase (total-cost) 100)
                 (usedGrasp ?o ?p ?g)
                 )
  )
  (:action clean
    :parameters (?o - wuti ?r - wuti)
    :precondition (and (stackable ?o ?r) (issink ?r)
                       (on ?o ?r)
                       (not (cleaned ?o))
                       )
    :effect (and (cleaned ?o)
                 (canpick)
                 (increase (total-cost) 100)
                 )
  )
  (:action cook
    :parameters (?o - wuti ?r - wuti)
    :precondition (and (stackable ?o ?r) (isstove ?r)
                       (on ?o ?r) (cleaned ?o))
    :effect (and (cooked ?o)
                 (not (cleaned ?o))
                 (canpick)
                 (increase (total-cost) 100)
                 )
  )


  (:derived (on ?ou - wuti ?os - wuti)
    (or (exists (?pu) (and (issupport ?ou ?pu ?os) (atpose ?ou ?pu)))
        (exists (?pu ?ps) (and (isstacksupport ?ou ?pu ?os ?ps) (atpose ?ou ?pu) (atpose ?os ?ps)))
    )
  )

  (:derived (occupied ?os - wuti)
    (exists (?ou) (on ?ou ?os))
  )

  (:derived (holding ?o - wuti)
    (exists (?g) (and (isgrasp ?o ?g)
                         (atgrasp ?o ?g)))
  )
)