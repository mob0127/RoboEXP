;; Xiao : 新增文件，用于本仓库对上游的扩展。
(define (domain pr2-pick-place)
  (:requirements :strips :equality)

  (:predicates
    ; Type membership (untyped domain because PDDLStream rejects :types)
    (Robot ?r)
    (IsObject ?o)
    (Surface ?s)
    (BConf ?q)
    (AConf ?q)
    (Pose ?p)
    (Grasp ?o ?g)
    (Motion ?path)

    ; Fluent state
    (RobotAt ?r ?q)
    (ArmAt ?r ?q)
    (ObjectAt ?o ?p)
    (Holding ?r ?o)
    (HandEmpty ?r)
    (On ?o ?s)

    ; Geometric constraints (certified by streams)
    (KinPick ?r ?o ?g ?bq ?aq ?p)
    (KinPlace ?r ?o ?g ?bq ?aq ?p)
    (BaseMotion ?q1 ?q2 ?path)
    (Supported ?o ?s ?p)
  )

  (:action move_base
    :parameters (?r ?q1 ?q2 ?path)
    :precondition (and (Robot ?r) (BConf ?q1) (BConf ?q2) (Motion ?path)
                       (RobotAt ?r ?q1) (BaseMotion ?q1 ?q2 ?path))
    :effect (and (RobotAt ?r ?q2)
                 (not (RobotAt ?r ?q1)))
  )

  (:action move_arm
    :parameters (?r ?q1 ?q2)
    :precondition (and (Robot ?r) (AConf ?q1) (AConf ?q2)
                       (ArmAt ?r ?q1))
    :effect (and (ArmAt ?r ?q2)
                 (not (ArmAt ?r ?q1)))
  )

  (:action pick
    :parameters (?r ?o ?g ?bq ?aq ?p)
    :precondition (and (Robot ?r) (IsObject ?o) (Grasp ?o ?g)
                       (BConf ?bq) (AConf ?aq) (Pose ?p)
                       (RobotAt ?r ?bq)
                       (ArmAt ?r ?aq)
                       (ObjectAt ?o ?p)
                       (HandEmpty ?r)
                       (KinPick ?r ?o ?g ?bq ?aq ?p))
    :effect (and (Holding ?r ?o)
                 (not (ObjectAt ?o ?p))
                 (not (HandEmpty ?r))))

  (:action place
    :parameters (?r ?o ?g ?bq ?aq ?p ?s)
    :precondition (and (Robot ?r) (IsObject ?o) (Grasp ?o ?g)
                       (Surface ?s) (BConf ?bq) (AConf ?aq) (Pose ?p)
                       (RobotAt ?r ?bq)
                       (ArmAt ?r ?aq)
                       (Holding ?r ?o)
                       (KinPlace ?r ?o ?g ?bq ?aq ?p)
                       (Supported ?o ?s ?p))
    :effect (and (ObjectAt ?o ?p)
                 (HandEmpty ?r)
                 (On ?o ?s)
                 (not (Holding ?r ?o))))
)
